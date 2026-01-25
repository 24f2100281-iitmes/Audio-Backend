from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import numpy as np
import librosa
import soundfile as sf

# ---------------- APP SETUP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # same as before
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- UTILS ----------------
def save_upload(file: UploadFile):
    path = os.path.join(TEMP_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path

# ---------------- ENCODE ----------------
def encode_audio(original_path, watermark_path):
    orig, sr1 = librosa.load(original_path, sr=None, mono=True)
    wm, sr2 = librosa.load(watermark_path, sr=None, mono=True)

    # Match sample rate
    if sr1 != sr2:
        wm = librosa.resample(wm, sr2, sr1)

    wm_length = len(wm)  # ⭐ STORE ORIGINAL LENGTH

    # Match length
    if len(wm) > len(orig):
        wm = wm[:len(orig)]
    else:
        wm = np.pad(wm, (0, len(orig) - len(wm)))

    # FFT
    FFT_orig = np.fft.fft(orig)
    FFT_wm = np.fft.fft(wm)

    alpha = 0.008
    N = len(FFT_orig)
    start = int(0.7 * N)

    FFT_watermarked = FFT_orig.copy()
    FFT_watermarked[start:] += alpha * FFT_wm[start:]

    watermarked_audio = np.real(np.fft.ifft(FFT_watermarked))
    watermarked_audio /= np.max(np.abs(watermarked_audio))

    output_file = "watermarked_audio_fft.wav"
    output_path = os.path.join(TEMP_DIR, output_file)
    sf.write(output_path, watermarked_audio, sr1)

    # ⭐ Save watermark length
    np.save(os.path.join(TEMP_DIR, "wm_length.npy"), wm_length)

    return output_file

# ---------------- DECODE ----------------
def decode_audio(original_path, watermarked_path):
    orig, sr1 = librosa.load(original_path, sr=None, mono=True)
    watermarked, sr2 = librosa.load(watermarked_path, sr=None, mono=True)

    # Match sample rate
    if sr1 != sr2:
        watermarked = librosa.resample(watermarked, sr2, sr1)

    min_len = min(len(orig), len(watermarked))
    orig = orig[:min_len]
    watermarked = watermarked[:min_len]

    FFT_orig = np.fft.fft(orig)
    FFT_watermarked = np.fft.fft(watermarked)

    alpha = 0.008
    N = len(FFT_orig)
    start = int(0.7 * N)

    FFT_extracted = np.zeros(N, dtype=complex)
    FFT_extracted[start:] = (FFT_watermarked[start:] - FFT_orig[start:]) / alpha

    extracted = np.real(np.fft.ifft(FFT_extracted))

    # ⭐ Exact trimming using saved length
    wm_length_path = os.path.join(TEMP_DIR, "wm_length.npy")
    if not os.path.exists(wm_length_path):
        raise ValueError("Watermark length metadata not found")

    wm_length = int(np.load(wm_length_path))
    extracted = extracted[:wm_length]

    if np.max(np.abs(extracted)) > 0:
        extracted /= np.max(np.abs(extracted))

    output_file = "extracted_watermark.wav"
    output_path = os.path.join(TEMP_DIR, output_file)
    sf.write(output_path, extracted, sr1)

    return output_file

# ---------------- API ROUTES ----------------
@app.post("/encode")
async def encode(
    original: UploadFile = File(...),
    watermark: UploadFile = File(...)
):
    orig_path = save_upload(original)
    wm_path = save_upload(watermark)

    output_filename = encode_audio(orig_path, wm_path)

    return {
        "message": "Encoding successful",
        "download_url": f"/download/{output_filename}"
    }

@app.post("/decode")
async def decode(
    original: UploadFile = File(...),
    watermarked: UploadFile = File(...)
):
    orig_path = save_upload(original)
    wm_path = save_upload(watermarked)

    output_filename = decode_audio(orig_path, wm_path)

    return {
        "message": "Decoding successful",
        "download_url": f"/download/{output_filename}"
    }

@app.get("/download/{filename}")
def download(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(TEMP_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    background_tasks.add_task(os.remove, file_path)

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )
