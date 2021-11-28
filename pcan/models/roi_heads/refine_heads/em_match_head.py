import math
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
from PIL import Image
import matplotlib.pyplot as plt

def pos_embed(x, temperature=10000, scale=2 * math.pi, normalize=False):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work on
    images.
    """
    batch_size, channel, height, width = x.size()
    mask = x.new_ones((batch_size, height, width))
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = channel // 2
    assert num_pos_feats * 2 == channel, (
        'The input channel number must be an even number.')
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

@HEADS.register_module
class EMMatchHeadPlus(nn.Module):
    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=80,
                 pos_proto_num=4,
                 neg_proto_num=4,
                 stage_num=6,
                 conv_cfg=None,
                 norm_cfg=None,
                 mask_thr_binary=0.5,
                 match_score_thr=0.5,
                 with_mask_ref=True,
                 with_mask_key=False,
                 with_both_feat=False,
                 with_dilation=False,
                 match_with_pids=True,
                 loss_mask=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.pos_proto_num = pos_proto_num
        self.neg_proto_num = neg_proto_num
        self.stage_num = stage_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.mask_thr_binary = mask_thr_binary
        self.match_score_thr = match_score_thr
        self.with_mask_ref = with_mask_ref
        self.with_mask_key = with_mask_key
        self.with_both_feat = with_both_feat
        self.with_dilation = with_dilation
        self.match_with_pids = match_with_pids
        self.loss_mask = build_loss(loss_mask)
        self.frame_num = 0

        padding = (self.conv_kernel_size - 1) // 2
        conv1_in_channels = self.in_channels + 2
        if self.with_mask_ref:
            conv1_in_channels += 1
        if self.with_mask_key:
            conv1_in_channels += 1
        self.conv1 = ConvModule(
            conv1_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2 = ConvModule(
            self.in_channels,
            self.conv_out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
    
        self.convs = nn.ModuleList()
        for i in range(1, self.num_convs):
            dilation = 2 ** i if self.with_dilation else 1
            padding = (self.conv_kernel_size - 1) // 2 * dilation
            self.convs.append(
                ConvModule(
                    self.conv_out_channels,
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

        self.init_protos(pos_proto_num, 'pos_mu')
        self.init_protos(neg_proto_num, 'neg_mu')

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
            if self.match_with_pids:
                same_pids = key_pids[i][:, None] == ref_pids[i][None, :]
                zeros = cos_dist.new_zeros(cos_dist.size())
                scores = torch.where(same_pids, cos_dist, zeros)
            else:
                scores = cos_dist

            conf, ref_ind = torch.max(scores, dim=1)
            valid = conf > self.match_score_thr
            ref_ind = ref_ind[valid]
            valids.append(valid)
            ref_inds.append(ref_ind)

        return valids, ref_inds

    def init_protos(self, proto_num, proto_name):
        protos = torch.Tensor(1, self.in_channels, proto_num)
        protos.normal_(0, math.sqrt(2. / proto_num))
        protos = self._l2norm(protos, dim=1)
        self.register_buffer(proto_name, protos)

    @force_fp32(apply_to=('inp', ))
    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @force_fp32(apply_to=('inp', ))
    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    @force_fp32(apply_to=('x', 'mask', 'mu'))
    @torch.no_grad()
    def _em_iter(self, x, mask, mu, time):
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)                                 # r * c * n
        m = mask.view(R, 1, H * W)                              # r * 1 * n
        if time == 0:
            mu = mu.repeat(R, 1, 1)                                 # r * c * k
        for _ in range(self.stage_num):
            z = torch.einsum('rcn,rck->rnk', (x, mu))           # r * n * k
            z = F.softmax(20 * z, dim=2)                        # r * n * k
            z = torch.einsum('rnk,ron->rnk', (z, m))            # r * n * k
            z = self._l1norm(z, dim=1)                          # r * n * k
            mu = torch.einsum('rcn,rnk->rck', (x, z))           # r * c * k
            mu = self._l2norm(mu, dim=1)                        # r * c * k
        return mu
  
    @force_fp32(apply_to=('x', 'mu'))
    def _prop(self, x, mu):
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)
        z = torch.einsum('rck,rcn->rkn', (mu, x))               # r * k * n
        z = F.softmax(z, dim=1)                                 # r * k * n
        
        pos_z, neg_z = z.chunk(2, dim=1)                        # r * k * n
        
        pos_z_c = pos_z.clone()
        neg_z_c = neg_z.clone()
        pos_z_c = pos_z_c.view(R, -1, H, W)               # r * 1 * h * w
        neg_z_c = neg_z_c.view(R, -1, H, W)               # r * 1 * h * w

        pos_z, neg_z = pos_z.contiguous(), neg_z.contiguous()
        pos_z = pos_z.sum(dim=1).view(R, 1, H, W)               # r * 1 * h * w
        neg_z = neg_z.sum(dim=1).view(R, 1, H, W)               # r * 1 * h * w
        return pos_z, neg_z, pos_z_c, neg_z_c

    def em_match(self, feat_a, mask_a, feat_b, mask_b):
        if not self.with_both_feat:
            pos_mu_pre = self._em_iter(feat_a, mask_a, self.pos_mu, 0)
            neg_mu_pre = self._em_iter(feat_a, 1 - mask_a, self.neg_mu, 0)
            pos_mu = self._em_iter(feat_b, mask_b, pos_mu_pre, 1)
            neg_mu = self._em_iter(feat_b, 1 - mask_b, neg_mu_pre, 1)
        else:
            feat = torch.cat((feat_a, feat_b), dim=2)
            mask = torch.cat((mask_a, mask_b), dim=2)
            pos_mu = self._em_iter(feat, mask, self.pos_mu)
            neg_mu = self._em_iter(feat, 1 - mask, self.neg_mu)

        mu = torch.cat((pos_mu, neg_mu), dim=2)
        pos_mu_c = pos_mu.clone()
        neg_mu_c = neg_mu.clone()
        pos_z, neg_z, pos_z_c, neg_z_c = self._prop(feat_b, mu)
        R = feat_b.size(0)
        pos_mu = pos_mu.permute(0, 2, 1).contiguous().view(R, -1, 1, 1)
        return pos_mu, pos_z, neg_z, pos_mu_c, neg_mu_c, pos_z_c, neg_z_c

    def compute_context(self, feat, mask, eps=1e-5):
        R, C, H, W = feat.size()

        fore_feat = (feat * mask).view(R, C, -1).sum(dim=2)
        fore_sum = mask.view(R, 1, -1).sum(dim=2)
        fore_feat = fore_feat / (fore_sum + eps)
        fore_feat = fore_feat.view(R, C, 1, 1)

        return fore_feat

    def gather_context(self, feat, gap):
        res = self.conv1(feat) + self.conv2(gap)
        return res

    @auto_fp16()
    def forward(self, feat_a, mask_a, feat_b, mask_b):
        feat_a_ori = feat_a.clone()
        feat_b_ori = feat_b.clone()

        mask_a = F.interpolate(mask_a, size=feat_a.size()[-2:], mode='bilinear')
        mask_b = F.interpolate(mask_b, size=feat_b.size()[-2:], mode='bilinear')

        m_a = mask_a.clone()
        m_b = mask_b.clone()

        mask_a = (mask_a.sigmoid() >= 0.5).float()
        mask_b = (mask_b.sigmoid() >= 0.5).float()

        gap = self.compute_context(feat_a_ori, mask_a)
        pos_mu, pos_z, neg_z, pos_mu_c, neg_mu_c, pos_z_c, neg_z_c = self.em_match(feat_a, mask_a, feat_b, mask_b)
        
        feat_cat = [feat_b_ori, pos_z, neg_z]
        if self.with_mask_ref:
            feat_cat.append(m_a)
        if self.with_mask_key:
            feat_cat.append(m_b)
        feat = torch.cat(feat_cat, dim=1)
        x = self.gather_context(feat, gap)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)

        mask_pred = self.conv_logits(x)
        return mask_pred 

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
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            segm = im_mask[i].cpu().numpy()
            cls_segms[labels[i]].append(segm)
            segms.append(segm)
        return cls_segms, segms
