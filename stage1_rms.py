"""
Stage 1 — Mic input, print RMS every 20ms.
Run this, speak, watch numbers go up. Ctrl+C to stop.
"""

import queue
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
BLOCK_SIZE  = 320        # 20ms at 16kHz
audio_q     = queue.Queue()


def callback(indata, frames, time, status):
    # This runs on the audio thread — only put, never block
    audio_q.put(bytes(indata))


def main():
    print("Listening... speak into your mic. Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "int16",
        blocksize  = BLOCK_SIZE,
        callback   = callback,
    ):
        while True:
            frame = audio_q.get()
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples ** 2))
            bar = "█" * int(rms / 200)
            print(f"RMS: {rms:6.1f}  {bar}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")