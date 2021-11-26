import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS

from .fcn_mask_head_plus import FCNMaskHeadPlus


@HEADS.register_module()
class FeatFCNMaskHeadPlus(FCNMaskHeadPlus):
    """Also return features before the last conv.
    """

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        feat = x
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred, feat
