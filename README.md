# SAM4SS: Customize **[SAM](https://github.com/facebookresearch/segment-anything)** and **[SAM2](https://github.com/facebookresearch/segment-anything-2)** for **Semantic Segmentation**

## Installation

```
git clone https://github.com/zhengxuJosh/SAM4SS.git
cd SAM4SS
conda create -n sam python==3.10.0
pip install -r requirements.txt
```

Install SAM2 and ideally using CUDA 12.x and cuDNN 9.x.x

```
cd ./SAM4SS/SAM2
pip install -e .
```

## TODO

- [x] Adopt SAM and SAM2 for multi-class image semantic segmentation predictions
- [ ] Conduct experiments on mainstream semantic segmentation datasets with full fine-tuning on SAM and SAM2
- [ ] Conduct experiments on mainstream semantic segmentation datasets with Parameter-Efficient Fine-Tuning (PEFT) on SAM and SAM2



## Getting Started

### Download Checkpoints

Please follow the official steps in [SAM](https://github.com/facebookresearch/segment-anything) and [SAM2](https://github.com/facebookresearch/segment-anything-2) to download the checkpoints.

### Try Demo

```
python3 ./SAM/demo.py
python3 ./SAM2/demo.py
```

#### Image Semantic Segmentation with SAM

Initialize the SAM model by giving input image size `image_size`, number of classes `num_classes` for your semantic segmentation dataset and also load the downloaded weights `checkpoint`.

```
sam, img_embedding_size = sam_model_registry['vit_h'](
    image_size=1024,
    num_classes=25,
    checkpoint='./sam_vit_h_4b8939.pth',
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1])
```

Forward propagation.

```
output = net(batched_input=input, multimask_output=True, image_size=1024)
```

#### Image Semantic Segmentation with SAM2

Using high resolution features (different from SAM which only using the image embeddings) and the image embeddings for generating masks.

```
import torch.nn as nn
from .modeling.sam2_base import SAM2Base

class sam4ss(nn.Module):

    def __init__(self, sam_model: SAM2Base):
        super(sam4ss, self).__init__()

        self.sam = sam_model

    def forward(self, batched_input, multimask_output=True):
            image_embedding = self.sam.forward_image(batched_input)

            return self.sam._forward_sam_heads(image_embedding['vision_features'], high_res_features=image_embedding['backbone_fpn'][:2], multimask_output=multimask_output)
```

Initialize the SAM2 model with corresponding config files and also load the pretrained weights.

```
checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "./sam2_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, checkpoint)
net = sam4ss(sam2)
```

Forward propagation.

```
output = net(batched_input=input, multimask_output=True)
```
