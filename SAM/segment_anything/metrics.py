import torch
from torch import Tensor
from typing import Tuple


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    # def update(self, pred: Tensor, target: Tensor) -> None:
    #     pred = pred.argmax(dim=1)
    #     print(self.ignore_label)
    #     print(target.shape,pred.shape)
    #     keep = target != self.ignore_label
    #     self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        print(f"Ignore label: {self.ignore_label}")
        print(f"Target shape: {target.shape}, Pred shape: {pred.shape}")
        
        keep = target != self.ignore_label
        
        target_kept = target[keep].view(-1)  # Flatten the kept target tensor
        pred_kept = pred[keep].view(-1)      # Flatten the kept pred tensor
        
        indices = target_kept * self.num_classes + pred_kept
        
        # Debugging prints
        print(f"Target kept min: {target_kept.min()}, max: {target_kept.max()}")
        print(f"Pred kept min: {pred_kept.min()}, max: {pred_kept.max()}")
        print(f"Indices min: {indices.min()}, max: {indices.max()}")
        
        # Clamping indices to ensure they are within the range [0, self.num_classes**2 - 1]
        indices = indices.clamp(0, self.num_classes**2 - 1)
        
        bincount_result = torch.bincount(indices, minlength=self.num_classes**2)
        
        if bincount_result.numel() == self.num_classes**2:
            self.hist += bincount_result.view(self.num_classes, self.num_classes)
        else:
            raise ValueError(f"Expected bincount result to have {self.num_classes**2} elements, but got {bincount_result.numel()}")



    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()]=0.
        miou = ious.mean().item()
        # miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()]=0.
        mf1 = f1.mean().item()
        # mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()]=0.
        macc = acc.mean().item()
        # macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

