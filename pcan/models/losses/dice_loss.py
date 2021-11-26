import torch
import torch.nn as nn

from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def dice_loss(pred, label, eps=1e-3):
    pred = pred.sigmoid().contiguous().view(pred.size(0), -1)
    label = label.contiguous().view(label.size(0), -1).float()

    a = torch.sum(pred * label, dim=1)
    b = torch.sum(pred * pred, dim=1) + eps
    c = torch.sum(label * label, dim=1) + eps
    d = (2 * a) / (b + c)
    return 1 - d


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert cls_score.size() == label.size()
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * dice_loss(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls

