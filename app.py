"""Local web app: drag-and-drop 4x upscaling in the browser, inference on this machine's GPU.

    python app.py            # opens http://127.0.0.1:7860
"""
import io
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from upscaler import Upscaler

HERE = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 7860

app = FastAPI(title="AI Upscaler")
upscaler = Upscaler()
lock = threading.Lock()  # one image on the GPU at a time


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {"device": upscaler.device_name}


@app.post("/api/upscale")
def upscale(file: UploadFile = File(...), dpi: int = Form(300), tile: int = Form(512)):
    try:
        im = Image.open(file.file)
        im.load()
    except Exception:
        raise HTTPException(400, "Not a valid image")
    with lock:
        result = upscaler.upscale(im, tile)
    buf = io.BytesIO()
    result.save(buf, "PNG", dpi=(dpi, dpi))
    stem = Path(file.filename or "image").stem
    return Response(buf.getvalue(), media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="{stem}_x4.png"',
        "X-Image-Size": f"{result.width}x{result.height}",
    })


if __name__ == "__main__":
    print(f"Device: {upscaler.device_name}  ->  http://{HOST}:{PORT}")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT)
