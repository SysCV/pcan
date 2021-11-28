import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops import Conv2d
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
    _do_paste_mask, BYTES_PER_FLOAT, GPU_MEM_LIMIT)

from mmdet.core import mask_target
from .boundary import get_instances_contour_interior

from pytorch_toolbelt import losses as L

@HEADS.register_module()
class BoundFCNMaskHeadPlus(FCNMaskHead):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=8,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(BoundFCNMaskHeadPlus, self).__init__(num_classes=8)
        assert(num_classes==8)
        self.convs_b = nn.ModuleList()
        self.num_classes = num_classes
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_b.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()

        if self.upsample_method is None:
            self.upsample_b = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample_b = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits_b = Conv2d(logits_in_channel, out_channels, 1)

    @auto_fp16()
    def forward(self, x):
        x_b = x.clone()
        for conv_b in self.convs_b:
            x_b = conv_b(x_b)

        x += x_b

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)

        if self.upsample_b is not None:
            x_b = self.upsample_b(x_b)
            if self.upsample_method == 'deconv':
                x_b = self.relu(x_b)
        bound_pred = self.conv_logits_b(x_b)
        return mask_pred, bound_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)

        #print('loss mask:', loss_mask.shape)
        loss['loss_mask'] = loss_mask
        return loss

    @force_fp32(apply_to=('bound_pred', ))
    def loss_bound(self, bound_pred, bound_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_bound = self.loss_mask(bound_pred, bound_targets,
                                       torch.zeros_like(labels))
        else:
            bound_targets = bound_targets.unsqueeze(1).repeat(1, self.num_classes, 1, 1)
            loss_bound = L.JointLoss(L.BceLoss(), L.BceLoss())(
                bound_pred.unsqueeze(1), bound_targets.to(dtype=torch.float32))


        loss['loss_bound'] = loss_bound * 0.5
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        boundary_ls = []
        for mask in mask_targets:
            mask_b = mask.data.cpu().numpy()
            boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
            boundary = torch.from_numpy(boundary).float().unsqueeze(0)

            boundary_ls.append(boundary)

        bound_targets = torch.cat(boundary_ls, dim=0).to(device=mask_targets.device)
       
        return mask_targets, bound_targets

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        FCNMaskHead returns a list of segms for each class, while this 'plus'
        version also returns a list of whole segms.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        segms = []
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            segm = im_mask[i].cpu().numpy()
            cls_segms[labels[i]].append(segm)
            segms.append(segm)
        return cls_segms, segms
