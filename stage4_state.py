"""
Stage 4 — State machine.

States:
    IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTING → LISTENING

The state machine is the single source of truth.
Nothing acts unless the state allows it.

Run this, press Enter to simulate the full bot turn cycle.
Speak during SPEAKING state to trigger barge-in.
"""

import queue
import threading
import collections
import time
from enum import Enum, auto
import numpy as np
import sounddevice as sd
import webrtcvad

# ── constants ──────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16_000
BLOCK_SIZE     = 320
RING_SIZE      = 5
SPEECH_TRIGGER = 3

# ── state machine ──────────────────────────────────────────────────────────

class State(Enum):
    IDLE         = auto()
    LISTENING    = auto()
    THINKING     = auto()
    SPEAKING     = auto()
    INTERRUPTING = auto()


# valid transitions — what each state is allowed to move to
TRANSITIONS = {
    State.IDLE:         {State.LISTENING},
    State.LISTENING:    {State.THINKING, State.IDLE},
    State.THINKING:     {State.SPEAKING, State.INTERRUPTING},
    State.SPEAKING:     {State.INTERRUPTING, State.LISTENING},
    State.INTERRUPTING: {State.LISTENING},
}


class StateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._lock  = threading.Lock()   # two threads touch state — needs a lock

    @property
    def state(self) -> State:
        return self._state

    def transition(self, to: State):
        with self._lock:
            allowed = TRANSITIONS.get(self._state, set())
            if to not in allowed:
                print(f"  [state] INVALID: {self._state.name} → {to.name} (ignored)")
                return False
            print(f"  [state] {self._state.name} → {to.name}")
            self._state = to
            return True

    def is_interruptible(self) -> bool:
        return self._state in (State.SPEAKING, State.THINKING)


# ── shared objects ─────────────────────────────────────────────────────────

audio_q         = queue.Queue()
interrupt_event = threading.Event()
sm              = StateMachine()


# ── mic callback ───────────────────────────────────────────────────────────

def callback(indata, frames, time, status):
    audio_q.put(bytes(indata))


# ── VAD thread ─────────────────────────────────────────────────────────────

def vad_thread():
    vad       = webrtcvad.Vad(3)
    ring      = collections.deque(maxlen=RING_SIZE)
    in_speech = False

    while True:
        frame     = audio_q.get()
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        ring.append(is_speech)

        if len(ring) < RING_SIZE:
            continue

        speech_frames = sum(ring)

        if not in_speech and speech_frames >= SPEECH_TRIGGER:
            in_speech = True

            # only interrupt if the bot is in an interruptible state
            if sm.is_interruptible():
                sm.transition(State.INTERRUPTING)
                interrupt_event.set()

        elif in_speech and speech_frames < SPEECH_TRIGGER:
            in_speech = False


# ── fake bot turn ──────────────────────────────────────────────────────────

def fake_thinking(duration: float = 1.5):
    """Simulates LLM thinking — just a sleep."""
    sm.transition(State.THINKING)
    print("  [bot] thinking...")

    start = time.monotonic()
    while time.monotonic() - start < duration:
        if interrupt_event.is_set():
            print("  [bot] interrupted during thinking")
            return False       # signal: was interrupted
        time.sleep(0.05)

    return True                # signal: finished naturally


def fake_speaking(duration: float = 5.0, freq: float = 440.0):
    """Plays sine wave. Aborts immediately on barge-in."""
    sm.transition(State.SPEAKING)
    print("  [bot] speaking... (talk to interrupt)\n")

    total_samples  = int(SAMPLE_RATE * duration)
    samples_played = 0

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while samples_played < total_samples:

            if interrupt_event.is_set():
                stream.abort()
                print("  [bot] playback aborted\n")
                return False

            t     = np.arange(BLOCK_SIZE) / SAMPLE_RATE
            offset = samples_played / SAMPLE_RATE
            chunk = (np.sin(2 * np.pi * freq * (t + offset)) * 32767 * 0.3).astype(np.int16)

            stream.write(chunk)
            samples_played += BLOCK_SIZE

    return True


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # start VAD thread
    t = threading.Thread(target=vad_thread, daemon=True)
    t.start()

    sm.transition(State.LISTENING)

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
        blocksize  = BLOCK_SIZE,
        callback   = callback,
    ):
        while True:
            input("Press Enter to simulate user utterance received...\n")

            interrupt_event.clear()

            # THINKING phase
            finished = fake_thinking(duration=1.5)
            if not finished:
                sm.transition(State.LISTENING)
                continue

            # SPEAKING phase
            finished = fake_speaking(duration=5.0)
            if not finished:
                sm.transition(State.LISTENING)
                continue

            # natural end
            sm.transition(State.LISTENING)
            print("  [bot] done, back to listening\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")