# Voice Bot Barge-In / Interruption Handler

A voice bot that lets you interrupt it mid-sentence — just like talking to a real person. Built from scratch using raw audio, no voice AI frameworks.

---

## What is Barge-In?

When you call a customer support line and the bot is talking, you can speak over it and it stops — that's barge-in. Most voice bot tutorials get this wrong. They mute the mic while the bot speaks, so you have to wait for it to finish before you can say anything. That's not a conversation, that's a queue.

Real barge-in means:
- Mic is **always open**, even while the bot is speaking
- The bot detects your voice mid-playback
- It cancels everything — LLM stream, TTS stream, audio buffer — and listens to you
- All of this happens in under 50ms

This project builds that from the ground up.

---

## Why Build This From Scratch?

Frameworks like Pipecat, Retell, and Vapi solve this for you. But if you use them without understanding what's underneath, you can't:
- Debug latency issues
- Tune for your environment
- Build something differentiated
- Understand why your bot sounds robotic

The plumbing is the product in voice AI. This project teaches you the plumbing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MIC (always open)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ raw PCM, every 20ms
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    VAD (Voice Activity Detector)             │
│  webrtcvad → ring buffer (8 frames) → speech/silence        │
│  energy floor: RMS > 300 only                               │
│  minimum duration: 300ms before barge-in fires              │
└──────────┬──────────────────────────┬───────────────────────┘
           │ user finished speaking   │ user speaks mid-playback
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────────────────┐
│   STT (Whisper)  │       │         BARGE-IN HANDLER         │
│  local, on CPU   │       │  interrupt_event.set()           │
│  ~1500ms         │       │  LLM cancel → TTS cancel →       │
└────────┬─────────┘       │  buffer flush → LISTENING        │
         │                 └──────────────────────────────────┘
         ▼
┌──────────────────┐
│   LLM (Gemini)   │
│  gemini-2.0-flash│
│  ~1300ms         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   TTS (gTTS)     │
│  Google HTTP     │
│  ~400ms          │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                   SPEAKER (pygame mixer)                      │
│   checks interrupt_event every 50ms                          │
│   mixer.music.stop() on barge-in                             │
└──────────────────────────────────────────────────────────────┘
```

---

## State Machine

The state machine is the single source of truth. Nothing acts unless the current state allows it. This prevents race conditions between the VAD thread and the main thread.

```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ startup
                         ▼
                    ┌─────────────┐
              ┌────►│  LISTENING  │◄────────────────────┐
              │     └──────┬──────┘                     │
              │            │ VAD: speech detected        │
              │            ▼                             │
              │     ┌─────────────┐                     │
              │     │  THINKING   │                     │
              │     └──────┬──────┘                     │
              │            │ LLM response ready          │
              │            ▼                             │
              │     ┌─────────────┐   barge-in    ┌─────────────────┐
              │     │  SPEAKING   │──────────────►│  INTERRUPTING   │
              │     └──────┬──────┘               └────────┬────────┘
              │            │ finished naturally             │
              └────────────┘               ────────────────┘
```

### Why a state machine?

Without it, the VAD thread has no context. It sees speech and fires an interrupt — always, regardless of what's happening. This causes:

- **Bug 1** — interrupting during THINKING before bot even starts speaking
- **Bug 2** — double interrupt — barge-in fires twice on the same playback
- **Bug 3** — interrupting during IDLE when there's nothing to cancel

The state machine fixes all three with one check:

```python
def is_interruptible(self):
    return self._state == State.SPEAKING
```

VAD asks "does an interrupt make sense right now?" before acting. If not — ignored.

---

## The VAD Pipeline

### Why 20ms frames?

The audio hardware gives you chunks of audio at fixed intervals. 20ms is the sweet spot:
- Short enough that detection feels instant to humans
- Long enough that webrtcvad has enough signal to make a reliable decision
- Below 10ms — not enough signal. Above 30ms — perceptible latency.

At 16kHz, 20ms = 320 samples = 640 bytes per frame.

### Why a ring buffer?

webrtcvad makes a per-frame binary decision with no memory. Raw output is jittery:

```
frame 1:  SPEECH
frame 2:  SILENCE   ← soft consonant mid-word
frame 3:  SPEECH
frame 4:  SILENCE
```

This would produce dozens of START/END events per second. The ring buffer looks at the last 8 frames and requires 6 to be speech before confirming:

```python
RING_SIZE      = 8
SPEECH_TRIGGER = 6
```

One noisy frame doesn't flip the state. You need sustained evidence.

### Why the 300ms minimum duration gate?

A cough is loud enough to clear the ring buffer threshold. It passes VAD. Without a duration gate, every cough interrupts the bot.

Real speech intent is sustained — you don't start talking and stop in 100ms. The 300ms gate distinguishes a cough from an actual utterance:

```python
MIN_SPEECH_MS = 300

duration_ms = (time.monotonic() - speech_start) * 1000
if duration_ms >= MIN_SPEECH_MS:
    interrupt_event.set()
```

### Why the energy floor?

webrtcvad can misclassify very quiet background noise as speech. An RMS check filters frames below a threshold before they even reach VAD:

```python
rms = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))
is_speech = vad.is_speech(frame, SAMPLE_RATE) if rms > 300 else False
```

RMS < 300 = almost certainly not speech. Skip VAD entirely.

---

## The Cancellation Chain

When barge-in fires, cancellation goes **upstream first**:

```
LLM stream cancel → TTS cancel → audio buffer flush → playback stop
```

Why upstream first? If you stop playback first and leave LLM/TTS running, you waste API tokens and compute generating a response nobody will hear.

In practice with gTTS (non-streaming), it looks like:

```python
# VAD thread
interrupt_event.set()

# main thread — checks before every API call
if interrupt_event.is_set():
    return None   # don't call Gemini

# speak() — checks every 50ms during playback
if interrupt_event.is_set():
    mixer.music.stop()   # stop() discards buffered audio immediately
    return False
```

`mixer.music.stop()` vs `mixer.music.fadeout()` — stop discards immediately, fadeout plays out. For barge-in you always want stop.

---

## Threading Model

Two threads + the audio hardware callback:

```
Audio hardware
     │ every 20ms
     ▼
callback()          ← sounddevice audio thread (managed by OS)
     │ queue.put()
     ▼
audio_q             ← thread-safe handoff (queue.Queue)
     │ queue.get()
     ▼
vad_thread          ← your VAD thread
     │ interrupt_event.set() or utterance_ready.set()
     ▼
main thread         ← STT → LLM → TTS → playback
```

### Why queue.Queue between callback and VAD?

The audio callback runs on the OS audio thread. You cannot do any slow work there — file I/O, network calls, even print. If you block the callback, the audio driver misses its next 20ms deadline and you get glitches or dropped frames.

`queue.Queue` is the handoff point. Callback puts and returns immediately. VAD thread consumes at its own pace. Thread-safe by design.

### Why threading.Event for interrupts?

`threading.Event` is the simplest signaling primitive between threads. One thread calls `.set()`, another calls `.is_set()`. No locks needed, no shared mutable state.

```python
interrupt_event = threading.Event()

# VAD thread
interrupt_event.set()

# main thread / speak()
if interrupt_event.is_set():
    # stop everything
```

---

## The Services

### STT — faster-whisper (local)

Whisper runs entirely on your machine. No API key, no cost, no data leaving your device.

The VAD thread records raw PCM while you speak. When you stop, that audio gets wrapped into a WAV file in memory and fed to Whisper:

```python
buf = io.BytesIO()           # file in memory, not on disk
with wave.open(buf, "wb") as wf:
    wf.writeframes(b"".join(pcm_frames))

model = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, _ = model.transcribe(tmp_path)
```

`tiny` model = 40MB, ~1500ms on CPU. `small` model = 500ms but larger download.

### LLM — Gemini 2.0 Flash

Gemini receives the transcript and returns a response. Conversation history is maintained so the bot has memory across turns:

```python
conversation.append({"role": "user",  "parts": [user_text]})
result = client_gemini.models.generate_content(model="gemini-2.0-flash", contents=user_text)
conversation.append({"role": "model", "parts": [result.text]})
```

### TTS — gTTS

gTTS calls Google Translate's TTS endpoint, returns an MP3, pygame plays it. Not streaming — full audio generates before playback starts. This is the main architectural difference from production systems.

---

## Latency Breakdown

### This project (free tier everything)

| Stage | Latency | Bottleneck |
|---|---|---|
| VAD confirmation | ~100ms | 8 frames × 20ms window |
| STT (Whisper tiny, CPU) | ~1500ms | Local CPU inference |
| LLM (Gemini free tier) | ~1300ms | Free tier rate limits |
| TTS (gTTS HTTP) | ~400ms | HTTP round trip + generation |
| **Total turn latency** | **~3200ms** | |
| **Barge-in latency** | **42-48ms** | Ring buffer + event signal |

### Production systems

| Company | Total latency | How |
|---|---|---|
| ElevenLabs Conversational | ~500ms | Streaming TTS, optimized pipeline |
| Retell AI | ~800ms | End to end voice agent |
| Vapi | ~700ms | Configurable LLM |
| Bland AI | ~800ms | Telephony focused |
| Daily / Pipecat | ~600ms | Open source, self hosted |
| **Human conversation** | **~200ms** | Turn taking response time |

### Why the gap?

Three things production systems do that this project doesn't:

**1. Streaming STT** — Deepgram and AssemblyAI return partial transcripts word by word while you're still speaking. They don't wait for silence to start transcribing.

**2. LLM first token → TTS** — production systems start TTS as soon as the first sentence arrives from the LLM, not after the full response.

**3. Streaming TTS** — Cartesia and ElevenLabs stream audio chunks as text comes in. First audio plays within 150ms of LLM first token.

The production pipeline looks like:

```
you speak → streaming STT → LLM token 1 → TTS chunk 1 → audio starts
                                 ↓
                           LLM token 2 → TTS chunk 2 → audio continues
```

Everything overlapping. Nothing waiting for the previous stage to finish.

**The barge-in latency of 42-48ms is already at production level.** That part of the architecture is solid. The total latency is a services problem, not an architecture problem.

---

## How to Get to 500ms Total Latency

Swap these three things, touch nothing else in the architecture:

| Current | Replace with | Latency saved |
|---|---|---|
| faster-whisper local | Deepgram streaming STT | ~1300ms |
| Gemini free tier | Groq (fastest LLM API) | ~800ms |
| gTTS | Cartesia streaming TTS | ~300ms |

The state machine, VAD pipeline, threading model, cancellation chain — all stays identical. The architecture is not the bottleneck.

---

## Setup

```bash
# install dependencies
pip install sounddevice numpy webrtcvad faster-whisper google-genai gtts pygame

# set your Gemini API key (free at aistudio.google.com, no credit card)
export GEMINI_API_KEY="your-key-here"

# run
python3 stage5_metrics.py
```

First run downloads the Whisper tiny model (~40MB). Cached after that.

## Running

Quick steps to run the demos on macOS (zsh):

1. Create and activate a virtualenv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install sounddevice numpy webrtcvad faster-whisper google-genai gtts pygame
```

3. Grant microphone access (macOS):

- Open System Settings → Privacy & Security → Microphone and enable access for your terminal app.

4. Set your Gemini API key (needed for the full pipeline in stage 5):

```bash
export GEMINI_API_KEY="your-key-here"
# or run a single command with the variable:
GEMINI_API_KEY="your-key-here" python3 stage5_metrics.py
```

5. Run individual stages (each is standalone):

```bash
# stage 1: verify microphone + RMS values
python3 stage1_rms.py

# stage 2: VAD demo, prints SPEECH START / END events
python3 stage2_vad.py

# stage 3: barge-in demo with fake TTS (sine wave)
python3 stage3_bargein.py

# stage 4: state machine tests / demo
python3 stage4_state.py

# stage 5: full pipeline (Whisper → Gemini → gTTS)
python3 stage5_metrics.py
```

6. Stop any running script with Ctrl+C.

Troubleshooting:
- No audio input/output: check macOS sound settings and microphone permissions.
- Install errors: ensure you are using the virtualenv and a supported Python (3.9+ recommended).
- If Whisper model downloads stall: check network and rerun the stage; the model caches after the first download.

---

## Project Structure

```
voice-interruption-handler/
  stage1_rms.py         — mic input, print RMS. proves audio is flowing
  stage2_vad.py         — wire in webrtcvad, print SPEECH START / END
  stage3_bargein.py     — fake TTS (sine wave) + barge-in interruption
  stage4_state.py       — state machine controlling everything
  stage5_metrics.py     — full pipeline: Whisper + Gemini + gTTS + metrics
```

Each stage is a standalone runnable script. Build and verify each one before moving to the next. The concepts compound — don't skip ahead.

---

## Key Concepts Learned

**VAD** — voice activity detection. Binary per-frame decision: speech or silence. webrtcvad is the same library Google uses in Chrome.

**Ring buffer** — last N decisions, vote on the majority. Trades ~100ms onset latency for stable, flicker-free detection.

**Structured concurrency** — two threads coordinating through events and queues, not shared mutable state. The queue is always the boundary between the audio thread and your logic.

**State machine** — explicit states with valid transitions. Makes concurrent bugs impossible to hide — invalid transitions are rejected, not silently corrupted.

**Upstream cancellation** — cancel from source to sink, not sink to source. Stop generating before you stop playing.

**Audio buffer management** — `stop()` discards buffered audio immediately. `drain()` plays it out first. For barge-in, always stop.

---

## What This Powers in Production

Every real-time voice AI company has a version of this:

- **Retell AI** — outbound sales calls
- **Vapi** — voice agent platform
- **Bland AI** — enterprise telephony
- **ElevenLabs Conversational AI** — real-time voice agents
- **Daily / Pipecat** — open source voice pipeline

The barge-in handler is the piece that makes a voice bot feel like a conversation rather than a phone tree. It's not glamorous — it's a threading problem and an audio buffer problem. But it's what separates a demo from a product.

---

## What's Next

- **Streaming STT** — Deepgram WebSocket, get words back while still speaking
- **Streaming TTS** — Cartesia, play audio as tokens arrive from LLM  
- **Acoustic Echo Cancellation** — stop the mic from picking up the speaker
- **Speaker diarization** — distinguish your voice from background voices
- **WebRTC transport** — move from local audio to phone/browser calls# voice-interruption-handler
