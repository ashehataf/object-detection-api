
from PIL import Image
import torchvision.transforms as T
import io

transform = T.Compose([
    T.ToTensor()
])

def read_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return transform(image)

def postprocess(output, threshold=0.5):
    results = []
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score >= threshold:
            results.append({
                "box": box.tolist(),
                "label": int(label),
                "score": float(score)
            })
    return results
