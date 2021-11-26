import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
    _do_paste_mask, BYTES_PER_FLOAT, GPU_MEM_LIMIT)
from pcan.core import cal_similarity


@HEADS.register_module
class LocalMatchHeadPlus(nn.Module):
    def __init__(self,
                 window_sizes,
                 num_convs=4,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=80,
                 conv_cfg=None,
                 norm_cfg=None,
                 mask_thr_binary=0.5,
                 match_score_thr=0.5,
                 with_mask_ref=True,
                 with_mask_key=False,
                 with_dilation=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super().__init__()
        self.window_sizes = window_sizes
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.mask_thr_binary = mask_thr_binary
        self.match_score_thr = match_score_thr
        self.with_mask_ref = with_mask_ref
        self.with_mask_key = with_mask_key
        self.with_dilation = with_dilation
        self.loss_mask = build_loss(loss_mask)

        conv_in_channels = self.in_channels + 2 * len(self.window_sizes) + 2
        if self.with_mask_ref:
            conv_in_channels += 1
        if self.with_mask_key:
            conv_in_channels += 1
        padding = (self.conv_kernel_size - 1) // 2
        self.conv1 = ConvModule(
            conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2 = ConvModule(
            self.in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.convs = nn.ModuleList()
        for i in range(1, self.num_convs):
            conv_in_channels = self.conv_out_channels
            dilation = 2 ** i if self.with_dilation else 1
            padding = (self.conv_kernel_size - 1) // 2 * dilation
            self.convs.append(
                ConvModule(
                    conv_in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def match(self, key_embeds, ref_embeds, key_pids, ref_pids):
        num_imgs = len(key_embeds)

        valids, ref_inds = [], []
        for i in range(num_imgs):
            cos_dist = cal_similarity(
                key_embeds[i], ref_embeds[i], method='cosine')
            same_pids = key_pids[i][:, None] == ref_pids[i][None, :]
            zeros = cos_dist.new_zeros(cos_dist.size())
            scores = torch.where(same_pids, cos_dist, zeros)

            conf, ref_ind = torch.max(scores, dim=1)
            valid = conf > self.match_score_thr
            ref_ind = ref_ind[valid]
            valids.append(valid)
            ref_inds.append(ref_ind)

        return valids, ref_inds

    @auto_fp16()
    def forward(self, feat_a, mask_a, feat_b, mask_b):
        mask_a = F.interpolate(mask_a, size=feat_a.size()[-2:], mode='bilinear')
        mask_b = F.interpolate(mask_b, size=feat_b.size()[-2:], mode='bilinear')
        m_a = mask_a.clone()
        m_b = mask_b.clone()
        mask_a = (mask_a.sigmoid() > self.mask_thr_binary).float()
        mask_b = (mask_b.sigmoid() > self.mask_thr_binary).float()

        nl_cntx = self.global_match(feat_a, mask_a, feat_b)

        distance = self.compute_distance(feat_a, feat_b)
        distance = self.compute_matches(mask_a, distance)

        feat_cat = [feat_b, distance, nl_cntx]
        if self.with_mask_ref:
            feat_cat.append(m_a)
        if self.with_mask_key:
            feat_cat.append(m_b)
        x = torch.cat(feat_cat, dim=1)
        f = self.compute_context(feat_a, mask_a)
        x = self.conv1(x) + self.conv2(f)
        x = self.relu(x)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def compute_context(self, feat, mask, eps=1e-5):
        R, C, H, W = feat.size()

        fore_feat = (feat * mask).view(R, C, -1).sum(dim=2)
        fore_sum = mask.view(R, 1, -1).sum(dim=2)
        fore_feat = fore_feat / (fore_sum + eps)
        fore_feat = fore_feat.view(R, C, 1, 1)

        return fore_feat

    def global_match(self, feat, mask, x):
        R, C, H, W = feat.size()
        feat = feat.view(R, C, H * W)
        x = x.view(R, C, H * W)

        mask = mask.view(R, H * W, 1)
        mask_fore, mask_back = mask, 1 - mask

        dis_a2 = torch.einsum('bcm,bcm->bm', (feat, feat)).unsqueeze(dim=2)
        dis_b2 = torch.einsum('bcn,bcn->bn', (x, x)).unsqueeze(dim=1)
        dis_ab = torch.einsum('bcm,bcn->bmn', (feat, x))

        # [R, H * W, H * W]
        dis = (dis_a2 + dis_b2 - 2 * dis_ab)
        dis = 1 - 2 / (1 + torch.exp(dis))
        dis_fore = dis * mask_fore + 1 * mask_back
        dis_back = dis * mask_back + 1 * mask_fore
        dis_fore = torch.min(dis_fore, dim=1)[0]
        dis_back = torch.min(dis_back, dim=1)[0]
        dis = torch.stack((dis_fore, dis_back), dim=1)
        dis = dis.view(R, 2, H, W)

        return dis

    def compute_distance(self, feat_a, feat_b):
        R, C, H, W = feat_a.size()
        P = self.window_sizes[-1]
        K = 2 * P + 1

        unfolded_a = F.unfold(feat_a, kernel_size=K, padding=P)
        unfolded_a = unfolded_a.view(R, C, -1, H * W)
        feat_b = feat_b.view(R, C, H * W)

        dis_a2 = torch.einsum('bckn,bckn->bkn', (unfolded_a, unfolded_a))
        dis_b2 = torch.einsum('bcn,bcn->bn', (feat_b, feat_b)).unsqueeze(dim=1)
        dis_ab = torch.einsum('bckn,bcn->bkn', (unfolded_a, feat_b))

        # [R, K * K, H * W]
        dis = dis_a2 + dis_b2 - 2 * dis_ab
        dis = 1 - 2 / (1 + torch.exp(dis))

        # [R, K, K, H * W]
        dis = dis.view(R, K, K, H * W)

        dises = []
        for window_size in self.window_sizes[:-1]:
            pad = P - window_size
            cur_dis = dis[:, pad:-pad, pad:-pad].contiguous()
            cur_dis = cur_dis.view(R, -1, H * W)
            dises.append(cur_dis)
        dis = dis.view(R, -1, H * W)
        dises.append(dis)

        return dises

    def compute_matches(self, mask, dises):
        R, _, H, W = mask.size()
        P = self.window_sizes[-1]
        K = 2 * P + 1

        # [R, K * K, H * W]
        unfolded_mask = F.unfold(mask, kernel_size=K, padding=P)
        unfolded_mask = unfolded_mask.view(R, K, K, H * W)

        unfolded_masks = []
        for window_size in self.window_sizes[:-1]:
            pad = P - window_size
            cur_unfold_mask = unfolded_mask[:, pad:-pad, pad:-pad].contiguous()
            cur_unfold_mask = cur_unfold_mask.view(R, -1, H * W)
            unfolded_masks.append(cur_unfold_mask)
        unfolded_mask = unfolded_mask.view(R, -1, H * W)
        unfolded_masks.append(unfolded_mask)

        res_dises = []
        for mask, dis in zip(unfolded_masks, dises):
            fore_mask, back_mask = mask, 1 - mask
            fore_dis = dis * fore_mask + 1 * back_mask
            back_dis = dis * back_mask + 1 * fore_mask

            fore_dis = torch.min(fore_dis, dim=1, keepdim=True)[0]
            back_dis = torch.min(back_dis, dim=1, keepdim=True)[0]
            res_dises.append(fore_dis)
            res_dises.append(back_dis)
        res_dis = torch.cat(res_dises, dim=1).view(R, -1, H, W)
        return res_dis

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, 1, h, w).
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
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
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
