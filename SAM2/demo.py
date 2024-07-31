import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2.sam4ss import sam4ss

checkpoint = "/mnt/dev-ssd-8T/zheng/DELIVER/semseg/models/sam2/checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "./sam2_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, checkpoint)
print(0)
net = sam4ss(sam2).cpu()

input = torch.randn(1,3,1024,1024)
output = net.forward(input, True)
print(0)