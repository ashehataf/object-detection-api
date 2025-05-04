
from pytriton.model_config import Tensor, ModelConfig
from pytriton.triton import Triton
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def infer_fn(inputs):
    images = torch.from_numpy(inputs[0])
    outputs = model(images)
    return [np.array([out["scores"].detach().numpy() for out in outputs])]

with Triton() as triton:
    triton.bind(
        model_name="object_detector",
        infer_func=infer_fn,
        inputs=[Tensor(dtype=np.float32, shape=(3, 224, 224))],
        outputs=[Tensor(dtype=np.float32, shape=(None,))]
    )
    triton.serve()
