import numpy as np
import librosa
import soundfile as sf
import uuid
import os

def encode_audio(original_path, watermark_path, alpha=0.008):
    orig, sr1 = librosa.load(original_path, sr=None, mono=True)
    wm, sr2 = librosa.load(watermark_path, sr=None, mono=True)

    if sr1 != sr2:
        wm = librosa.resample(wm, orig_sr=sr2, target_sr=sr1)

    if len(wm) > len(orig):
        wm = wm[:len(orig)]
    else:
        wm = np.pad(wm, (0, len(orig) - len(wm)))

    FFT_orig = np.fft.fft(orig)
    FFT_wm = np.fft.fft(wm)

    N = len(FFT_orig)
    start = int(0.7 * N)

    FFT_watermarked = FFT_orig.copy()
    FFT_watermarked[start:] += alpha * FFT_wm[start:]

    watermarked_audio = np.real(np.fft.ifft(FFT_watermarked))
    watermarked_audio /= np.max(np.abs(watermarked_audio))

    filename = f"watermarked_{uuid.uuid4().hex}.wav"
    output_path = os.path.join("temp", filename)

    sf.write(output_path, watermarked_audio, sr1)

    return filename
