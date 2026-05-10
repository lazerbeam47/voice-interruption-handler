"""
Stage 5 — Full voice bot with latency metrics.

Pipeline:
  mic → VAD → Whisper (local) → Gemini (LLM) → gTTS → speaker

Metrics tracked:
  - VAD latency        (speech start → utterance end)
  - STT latency        (utterance end → transcript ready)
  - LLM latency        (transcript → full response)
  - TTS latency        (response text → audio ready)
  - Barge-in latency   (speech detected → audio stopped)
  - Total turn latency (utterance end → first audio)
"""

import os
import io
import queue
import threading
import collections
import time
import wave
import tempfile
from enum import Enum, auto

import numpy as np
import sounddevice as sd
import webrtcvad
from google import genai
from gtts import gTTS
from pygame import mixer

# ── config ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "your-gemini-api-key-here")

SAMPLE_RATE      = 16_000
BLOCK_SIZE       = 320
RING_SIZE        = 8
SPEECH_TRIGGER   = 6
SILENCE_TRIGGER  = 3
MIN_SPEECH_MS    = 300     # minimum speech duration to trigger barge-in

client_gemini = genai.Client(api_key=GEMINI_API_KEY)
mixer.init()

# ── metrics ────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t_speech_start  = None   # when VAD first confirmed speech
        self.t_utterance_end = None   # when user stopped speaking
        self.t_stt_done      = None   # when transcript was ready
        self.t_llm_done      = None   # when LLM response was ready
        self.t_tts_done      = None   # when audio started playing
        self.t_bargein       = None   # when barge-in was detected
        self.t_aborted       = None   # when audio was aborted

    def report(self):
        print("\n  ── latency report ──────────────────────────")

        if self.t_speech_start and self.t_utterance_end:
            vad = (self.t_utterance_end - self.t_speech_start) * 1000
            print(f"  VAD confirmation latency : {vad:.0f}ms")

        if self.t_utterance_end and self.t_stt_done:
            stt = (self.t_stt_done - self.t_utterance_end) * 1000
            print(f"  STT latency              : {stt:.0f}ms")

        if self.t_stt_done and self.t_llm_done:
            llm = (self.t_llm_done - self.t_stt_done) * 1000
            print(f"  LLM latency              : {llm:.0f}ms")

        if self.t_llm_done and self.t_tts_done:
            tts = (self.t_tts_done - self.t_llm_done) * 1000
            print(f"  TTS latency              : {tts:.0f}ms")

        if self.t_utterance_end and self.t_tts_done:
            total = (self.t_tts_done - self.t_utterance_end) * 1000
            print(f"  ── total turn latency    : {total:.0f}ms")

        if self.t_bargein and self.t_aborted:
            bargein = (self.t_aborted - self.t_bargein) * 1000
            print(f"  Barge-in latency         : {bargein:.0f}ms")

        print("  ────────────────────────────────────────────\n")


metrics = Metrics()


# ── state machine ──────────────────────────────────────────────────────────

class State(Enum):
    IDLE         = auto()
    LISTENING    = auto()
    THINKING     = auto()
    SPEAKING     = auto()
    INTERRUPTING = auto()


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
        self._lock  = threading.Lock()

    @property
    def state(self):
        return self._state

    def transition(self, to: State):
        with self._lock:
            allowed = TRANSITIONS.get(self._state, set())
            if to not in allowed:
                return False
            print(f"  [state] {self._state.name} → {to.name}")
            self._state = to
            return True

    def is_interruptible(self):
        return self._state == State.SPEAKING   # only interrupt during speech


# ── shared objects ─────────────────────────────────────────────────────────

audio_q         = queue.Queue()
interrupt_event = threading.Event()
utterance_ready = threading.Event()
utterance_audio = []
sm              = StateMachine()
conversation    = []


# ── mic callback ───────────────────────────────────────────────────────────

def callback(indata, frames, time, status):
    audio_q.put(bytes(indata))


# ── VAD thread ─────────────────────────────────────────────────────────────

def vad_thread():
    vad           = webrtcvad.Vad(3)
    ring          = collections.deque(maxlen=RING_SIZE)
    in_speech     = False
    recording     = []
    speech_start  = None

    while True:
        frame     = audio_q.get()

        # energy floor — ignore very quiet frames
        rms = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))
        is_speech = vad.is_speech(frame, SAMPLE_RATE) if rms > 300 else False

        ring.append(is_speech)

        if len(ring) < RING_SIZE:
            continue

        speech_frames = sum(ring)

        # ── speech start ───────────────────────────────────────────────────
        if not in_speech and speech_frames >= SPEECH_TRIGGER:
            in_speech    = True
            speech_start = time.monotonic()
            recording    = []
            metrics.t_speech_start = speech_start

        # ── accumulate audio ───────────────────────────────────────────────
        if in_speech:
            recording.append(frame)

            # barge-in — only after MIN_SPEECH_MS of sustained speech
            if sm.is_interruptible():
                duration_ms = (time.monotonic() - speech_start) * 1000
                if duration_ms >= MIN_SPEECH_MS:
                    print("\n  [vad] barge-in detected")
                    metrics.t_bargein = time.monotonic()
                    sm.transition(State.INTERRUPTING)
                    interrupt_event.set()

        # ── speech end ─────────────────────────────────────────────────────
        if in_speech and speech_frames < SILENCE_TRIGGER:
            in_speech = False
            metrics.t_utterance_end = time.monotonic()

            if sm.state == State.LISTENING and recording:
                utterance_audio.clear()
                utterance_audio.extend(recording)
                utterance_ready.set()

            recording = []


# ── STT — local whisper ────────────────────────────────────────────────────

def transcribe(pcm_frames):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        os.system("pip install faster-whisper --break-system-packages")
        from faster_whisper import WhisperModel

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(pcm_frames))
    buf.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(buf.read())
        tmp_path = f.name

    print("  [stt] transcribing...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(tmp_path, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(tmp_path)

    metrics.t_stt_done = time.monotonic()
    return text


# ── LLM — Gemini ───────────────────────────────────────────────────────────

def get_response(user_text):
    conversation.append({"role": "user", "parts": [user_text]})
    print(f"  [llm] user: {user_text}")
    print(f"  [llm] thinking...")

    if interrupt_event.is_set():
        return None

    result = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_text,
    )

    response_text = result.text.strip()
    conversation.append({"role": "model", "parts": [response_text]})
    print(f"  [llm] response: {response_text}")

    metrics.t_llm_done = time.monotonic()
    return response_text


# ── TTS — gTTS ─────────────────────────────────────────────────────────────

def speak(text):
    sm.transition(State.SPEAKING)
    print("  [tts] generating audio...")

    if interrupt_event.is_set():
        return False

    tts = gTTS(text=text, lang="en", slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        tmp_path = f.name

    mixer.music.load(tmp_path)
    mixer.music.play()

    metrics.t_tts_done = time.monotonic()
    print("  [tts] speaking... (talk to interrupt)\n")

    # report latency as soon as audio starts
    metrics.report()

    while mixer.music.get_busy():
        if interrupt_event.is_set():
            mixer.music.stop()
            metrics.t_aborted = time.monotonic()
            print("  [tts] playback stopped")

            # report barge-in latency
            metrics.report()

            os.unlink(tmp_path)
            return False
        time.sleep(0.05)

    os.unlink(tmp_path)
    return True


# ── main loop ──────────────────────────────────────────────────────────────

def main():
    print("Voice bot starting... speak to begin.\n")
    print("(First run downloads Whisper tiny model ~40MB)\n")

    threading.Thread(target=vad_thread, daemon=True).start()
    sm.transition(State.LISTENING)

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
        blocksize  = BLOCK_SIZE,
        callback   = callback,
    ):
        while True:
            utterance_ready.wait()
            utterance_ready.clear()
            interrupt_event.clear()
            metrics.reset()

            # flush stale audio
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except:
                    break

            time.sleep(0.3)
            interrupt_event.clear()

            frames = list(utterance_audio)
            if not frames:
                continue

            # STT
            sm.transition(State.THINKING)
            text = transcribe(frames)
            if not text:
                sm.transition(State.LISTENING)
                continue

            if interrupt_event.is_set():
                sm.transition(State.LISTENING)
                continue

            # LLM
            response = get_response(text)
            if not response or interrupt_event.is_set():
                sm.transition(State.LISTENING)
                continue

            # TTS
            speak(response)

            sm.transition(State.LISTENING)
            print("  [bot] listening...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")