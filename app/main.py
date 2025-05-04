
from fastapi import FastAPI, File, UploadFile, Form
from model.model import load_model
from app.utils import read_image, postprocess
import torch

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(image: UploadFile = File(...), image_id: str = Form(...)):
    img_bytes = await image.read()
    img_tensor = read_image(img_bytes).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    results = postprocess(outputs)
    return {"image_id": image_id, "detections": results}
