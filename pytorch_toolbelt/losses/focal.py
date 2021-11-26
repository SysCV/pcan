from functools import partial

from torch.nn.modules.loss import _Loss

from .functional import focal_loss_with_logits

__all__ = ["BinaryFocalLoss", "FocalLoss"]


class BinaryFocalLoss(_Loss):
    def __init__(
        self,
        alpha=0.5,
        gamma=2,
        ignore_index=None,
        reduction="mean",
        reduced=False,
        threshold=0.5,
    ):
        """

        :param alpha:
        :param gamma:
        :param ignore_index:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(
                focal_loss_with_logits,
                alpha=None,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.focal_loss = partial(
                focal_loss_with_logits, alpha=alpha, gamma=gamma, reduction=reduction
            )

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += focal_loss_with_logits(
                cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha
            )
        return loss
