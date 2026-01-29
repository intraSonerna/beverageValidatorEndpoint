from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="Beverage Detector")

MODEL_PATH = "/models/best.engine"  # eller best.pt
model = YOLO(MODEL_PATH)

def img_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = model.predict(img, imgsz=1024, conf=0.25)
    det = results[0]

    img_out = det.plot()
    img_b64 = img_to_base64(img_out)

    return JSONResponse({
        "num_objects": len(det.boxes),
        "found_any": len(det.boxes) > 0,
        "image_base64": img_b64
    })
