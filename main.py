from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import os
import uvicorn

from processing.encode import encode_audio
from processing.decode import decode_audio

app = FastAPI()
os.makedirs("temp", exist_ok=True)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def save_upload(file: UploadFile):
    path = os.path.join("temp", file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path


@app.get("/")
def health():
    return {"status": "Backend is running"}


# ----------- ENCODE -----------
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


# ----------- DECODE -----------
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


# ----------- DOWNLOAD ----------
@app.get("/download/{filename}")
def download(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join("temp", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    background_tasks.add_task(os.remove, file_path)

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)
