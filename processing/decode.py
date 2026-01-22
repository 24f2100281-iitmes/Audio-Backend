import numpy as np
import librosa
import soundfile as sf
import uuid
import os

def trim_silence(audio, threshold=0.01, frame_length=2048, hop_length=512):
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_length]**2))
        for i in range(0, len(audio), hop_length)
    ])
    non_silent = np.where(energy > threshold)[0]

    if len(non_silent) > 0:
        cut = (non_silent[-1] + 2) * hop_length + frame_length
        return audio[:min(cut, len(audio))]
    return audio


def decode_audio(original_path, watermarked_path, alpha=0.008):
    orig, sr1 = librosa.load(original_path, sr=None, mono=True)
    wm, sr2 = librosa.load(watermarked_path, sr=None, mono=True)

    if sr1 != sr2:
        wm = librosa.resample(wm, orig_sr=sr2, target_sr=sr1)

    min_len = min(len(orig), len(wm))
    orig = orig[:min_len]
    wm = wm[:min_len]

    FFT_orig = np.fft.fft(orig)
    FFT_wm = np.fft.fft(wm)

    N = len(FFT_orig)
    start = int(0.7 * N)

    FFT_extracted = np.zeros(N, dtype=complex)
    FFT_extracted[start:] = (FFT_wm[start:] - FFT_orig[start:]) / alpha

    extracted = np.real(np.fft.ifft(FFT_extracted))
    extracted /= np.max(np.abs(extracted))
    extracted = trim_silence(extracted)

    filename = f"extracted_{uuid.uuid4().hex}.wav"
    output_path = os.path.join("temp", filename)

    sf.write(output_path, extracted, sr1)

    return filename
