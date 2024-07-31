from segment_anything import sam_model_registry
import torch
sam, img_embedding_size = sam_model_registry['vit_h'](
    image_size=1024,
    num_classes=25,
    checkpoint='./sam_vit_h_4b8939.pth',
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1])
net = sam
input = torch.randn(1,3,1024,1024)
output = net(input, True, 1024)
