
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model
