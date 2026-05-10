"""
Stage 3 — Fake TTS playback + barge-in.

Plays a 5 second sine wave (fake bot speaking).
VAD runs simultaneously on the mic.
When you speak — playback stops immediately.

This is the core barge-in mechanic in isolation.
"""

import queue
import threading
import collections
import time
import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE    = 16_000
BLOCK_SIZE     = 320
RING_SIZE      = 5
SPEECH_TRIGGER = 4

audio_q        = queue.Queue()
vad            = webrtcvad.Vad(3)

# This is the single signal between VAD thread and playback thread
interrupt_event = threading.Event()


# ── VAD thread ─────────────────────────────────────────────────────────────

def vad_thread():
    """
    Runs continuously. When speech is detected during playback,
    sets interrupt_event to signal playback to stop.
    """
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
            print(">>> SPEECH DETECTED — interrupting bot")
            interrupt_event.set()        # signal playback to stop

        elif in_speech and speech_frames < SPEECH_TRIGGER:
            in_speech = False


# ── mic callback ───────────────────────────────────────────────────────────

def callback(indata, frames, time, status):
    audio_q.put(bytes(indata))


# ── fake TTS playback ──────────────────────────────────────────────────────

def play_sine(duration_sec: float = 5.0, freq: float = 440.0):
    """
    Plays a sine wave for up to duration_sec seconds.
    Checks interrupt_event every 20ms chunk — stops immediately when set.
    """
    total_samples  = int(SAMPLE_RATE * duration_sec)
    chunk_size     = BLOCK_SIZE
    samples_played = 0

    print(f"Bot speaking... (interrupt me by talking)\n")

    with sd.OutputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
    ) as stream:

        while samples_played < total_samples:

            # check for barge-in before every chunk
            if interrupt_event.is_set():
                stream.abort()           # abort() discards buffered audio immediately
                print("<<< PLAYBACK STOPPED\n")
                return

            # generate next 20ms chunk of sine wave
            t = np.arange(chunk_size) / SAMPLE_RATE
            offset = samples_played / SAMPLE_RATE
            chunk = (np.sin(2 * np.pi * freq * (t + offset)) * 32767 * 0.3).astype(np.int16)

            stream.write(chunk)
            samples_played += chunk_size

    print("Bot finished speaking naturally.\n")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # start VAD thread — runs forever in background
    t = threading.Thread(target=vad_thread, daemon=True)
    t.start()

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
        blocksize  = BLOCK_SIZE,
        callback   = callback,
    ):
        while True:
            input("Press Enter to simulate bot speaking (then talk to interrupt)...\n")

            # reset before each playback
            interrupt_event.clear()

            play_sine(duration_sec=5.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")