# SAM4SS： 

## Customize **[SAM](https://github.com/facebookresearch/segment-anything)** and **[SAM2](https://github.com/facebookresearch/segment-anything-2)** for **Semantic Segmentation**

## Installation

```
git clone --
cd --
conda create --
```

Please install SAM2 on a GPU machine using 

```
pip install -e .
```

## Getting Started

## Download Checkpoints

Please follow the official steps in [SAM](https://github.com/facebookresearch/segment-anything) and [SAM2](https://github.com/facebookresearch/segment-anything-2) to download the checkpoints.

## Try Demo

```
python3 ./SAM/demo.py
python3 ./SAM2/demo.py
```

## Semantic Segmentation with SAM

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
