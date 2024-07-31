import torch.nn as nn
from .modeling.sam2_base import SAM2Base

class sam4ss(nn.Module):

    def __init__(self, sam_model: SAM2Base):
        super(sam4ss, self).__init__()

        self.sam = sam_model

    def forward(self, batched_input, multimask_output=True):
            image_embedding = self.sam.forward_image(batched_input)

            return self.sam._forward_sam_heads(image_embedding['vision_features'], high_res_features=image_embedding['backbone_fpn'][:2], multimask_output=multimask_output) 
