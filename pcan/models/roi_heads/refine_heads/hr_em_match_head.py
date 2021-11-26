import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import ConvModule, build_upsample_layer
from mmdet.core import bbox_rescale
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import (
    _do_paste_mask, BYTES_PER_FLOAT, GPU_MEM_LIMIT)
from pcan.core import cal_similarity


def gen_pos_emb(x, temperature=10000, scale=2 * math.pi, normalize=False):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work on
    images.
    """
    R, C, H, W = x.size()
    mask = x.new_ones((R, H, W))
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = C // 2
    assert num_pos_feats * 2 == C, (
        'The input channel number must be an even number.')
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
    return pos


@HEADS.register_module
class HREMMatchHeadPlus(nn.Module):
    """HR means high-resolution. This version refine the mask projecting to the
    1/4 or 1/8 size of the original input, instead of refining in the RoI level.
    """

    def __init__(self,
                 num_feats=3,
                 num_convs=4,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_channels=128,
                 out_channels=8,
                 num_classes=80,
                 feat_stride=8,
                 out_stride=4,
                 pos_proto_num=4,
                 neg_proto_num=4,
                 stage_num=6,
                 with_mask_key=True,
                 with_both_feat=False,
                 with_pos_emb=False,
                 match_score_thr=0.5,
                 rect_scale_factor=1.5,
                 upsample_cfg=dict(type='deconv'),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='DiceLoss',
                     loss_weight=1.0)):
        super().__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_feats = num_feats
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels
        self.out_channels = out_channels
        self.feat_stride = feat_stride
        self.out_stride = out_stride
        self.num_classes = num_classes
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = feat_stride // out_stride
        self.pos_proto_num = pos_proto_num
        self.neg_proto_num = neg_proto_num
        self.stage_num = stage_num
        self.with_mask_key = with_mask_key
        self.with_both_feat = with_both_feat
        self.with_pos_emb = with_pos_emb
        self.match_score_thr = match_score_thr
        self.rect_scale_factor = rect_scale_factor
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.positioanl_embeddings = None

        self.refines = nn.ModuleList()
        for i in range(self.num_feats):
            in_channels = self.in_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.refines.append(
                ConvModule(
                    self.in_channels, self.conv_channels, self.conv_kernel_size,
                    padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg))

        padding = (self.conv_kernel_size - 1) // 2
        self.conv1 = ConvModule(
            self.conv_channels, self.out_channels, self.conv_kernel_size,
            padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            self.conv_channels, self.out_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=None)
        self.conv3 = ConvModule(
            3, self.out_channels, self.conv_kernel_size, padding=padding,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.conv4 = ConvModule(
            self.conv_channels, self.out_channels, 1, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=None)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    self.out_channels, self.out_channels, self.conv_kernel_size,
                    padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg))

        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=self.out_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        self.conv_logits = nn.Conv2d(self.out_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.init_protos(pos_proto_num, 'pos_mu')
        self.init_protos(neg_proto_num, 'neg_mu')

    def pos_emb(self, x):
        if not self.with_pos_emb:
            return 0.
        if self.positioanl_embeddings is None:
            self.positioanl_embeddings = gen_pos_emb(x, normalize=True)
        return self.positioanl_embeddings

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

    def init_protos(self, proto_num, proto_name):
        protos = torch.Tensor(1, self.conv_channels, proto_num)
        protos.normal_(0, math.sqrt(2. / proto_num))
        protos = self._l2norm(protos, dim=1)
        self.register_buffer(proto_name, protos)

    @auto_fp16()
    def forward_feat(self, x):
        start_lvl = int(math.log2(self.feat_stride // 4))
        end_lvl = min(start_lvl + self.num_feats, len(x))
        feats = [
            refine(lvl)
            for refine, lvl in zip(self.refines, x[start_lvl:end_lvl])]
        for i in range(1, len(feats)):
            feats[i] = F.interpolate(feats[i], size=feats[0].size()[-2:],
                                     mode='bilinear')
        feat = sum(feats)
        # for conv in self.convs:
        #     feat = conv(feat)
        return feat

    @force_fp32(apply_to=('inp', ))
    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @force_fp32(apply_to=('inp', ))
    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    @force_fp32(apply_to=('feat', 'mask', 'mu'))
    @torch.no_grad()
    def _em_iter(self, feat, mask, mu):
        # n = h * w
        _, C = feat.size()[:2]
        R, _ = mask.size()[:2]

        pos_feat = feat + self.pos_emb(x=feat)
        x = pos_feat.view(1, C, -1)                             # 1 * C * N
        y = feat.view(1, C, -1)                                 # 1 * C * N
        m = mask.view(R, 1, -1)                                 # R * 1 * N
        mu = mu.repeat(R, 1, 1)                                 # R * C * K
        for i in range(self.stage_num):
            z = torch.einsum('ocn,rck->rnk', (x, mu))           # R * N * K
            z = F.softmax(z, dim=2)                             # R * N * K
            z = torch.einsum('rnk,ron->rnk', (z, m))            # R * N * K
            z = self._l1norm(z, dim=1)                          # R * N * K
            mu = torch.einsum('ocn,rnk->rck', (x, z))           # R * C * K
            mu = self._l2norm(mu, dim=1)                        # R * C * K
        nu = torch.einsum('ocn,rnk->rck', (y, z))               # R * C * K
        nu = self._l2norm(nu, dim=1)                            # R * C * K
        return mu, nu

    @force_fp32(apply_to=('feat', 'mu'))
    def _prop(self, feat, mu):
        R = mu.size(0)
        _, C, H, W = feat.size()
        pos_feat = feat + self.pos_emb(x=feat)
        x = pos_feat.view(1, C, -1)                             # 1 * C * N
        z = torch.einsum('rck,ocn->rkn', (mu, x))               # R * K * N
        z = F.softmax(z, dim=1)                                 # R * K * N
        z = z.view(R, 2, -1, H, W).sum(dim=2)                   # R * 2 * H * W
        return z

    def em_match(self, feat_a, mask_a, rect_a, feat_b, mask_b, rect_b):
        if not self.with_both_feat:
            pos_mask, neg_mask = rect_a * mask_a, rect_a * (1 - mask_a)
            pos_mu, pos_nu = self._em_iter(feat_a, pos_mask, self.pos_mu)
            neg_mu, neg_nu = self._em_iter(feat_a, neg_mask, self.neg_mu)
        else:
            feat = torch.cat((feat_a, feat_b), dim=2)
            mask = torch.cat((mask_a, mask_b), dim=2)
            rect = torch.cat((rect_a, rect_b), dim=2)
            pos_mask, neg_mask = rect * mask, rect * (1 - mask)
            pos_mu, pos_nu = self._em_iter(feat, pos_mask, self.pos_mu)
            neg_mu, neg_nu = self._em_iter(feat, neg_mask, self.neg_mu)
        mu = torch.cat((pos_mu, neg_mu), dim=2)
        z = self._prop(feat_b, mu)
        R = mask_b.size(0)
        pos_nu = pos_nu.permute(0, 2, 1).contiguous().view(R, -1, 1, 1)
        return pos_nu, z

    def compute_context(self, feat, mask, eps=1e-5):
        _, C = feat.size()[:2]
        R, _ = mask.size()[:2]

        fore_feat = (feat * mask).view(R, C, -1).sum(dim=2)
        fore_sum = mask.view(R, 1, -1).sum(dim=2)
        fore_feat = fore_feat / (fore_sum + eps)
        fore_feat = fore_feat.view(R, C, 1, 1)

        return fore_feat

    def gather_context(self, feat, mask, gap, z, pos_mu):
        mask = torch.cat((mask, z), dim=1)
        res = self.conv1(feat) + self.conv2(gap) + self.conv3(mask)
        res = res + F.conv2d(pos_mu, self.conv4.conv.weight,
                             self.conv4.conv.bias, groups=self.pos_proto_num)
        res = F.relu(res)
        return res

    @auto_fp16()
    def forward(self, x_a, mask_a, rect_a, x_b, mask_b, rect_b):
        assert len(mask_a) == len(mask_b) == x_a[0].size(0) == x_b[0].size(0)

        feat_a = self.forward_feat(x_a)
        feat_b = self.forward_feat(x_b)
        B, C, H, W = feat_a.size()

        feat_a = torch.chunk(feat_a, B, dim=0)
        feat_b = torch.chunk(feat_b, B, dim=0)

        xs = []
        for i in range(B):
            if len(mask_a[i]) == 0:
                continue
            m_a = mask_a[i].clone()
            m_b = mask_b[i].clone()
            mask_a[i] = mask_a[i].sigmoid()
            mask_b[i] = mask_b[i].sigmoid()
            # pos_mu: [R, K * C, 1, 1]
            # pos_z: [R, 1, H, W]
            pos_mu, z = self.em_match(feat_a[i], mask_a[i], rect_a[i],
                                      feat_b[i], mask_b[i], rect_b[i])
            # pos_feat: [R, C, 1, 1]
            gap = self.compute_context(feat_a[i], mask_a[i])
            # x: [R, C, H, W]
            mask = m_b if self.with_mask_key else m_a
            x = self.gather_context(feat_b[i], mask, gap, z, pos_mu)
            xs.append(x)

        x = torch.cat(xs, dim=0)
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, valids, gt_masks):
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds[valid]
            for res, valid in zip(sampling_results, valids)]
        mask_targets = map(self.get_target_single, pos_assigned_gt_inds,
                           gt_masks)
        mask_targets = list(mask_targets)
        if len(mask_targets) > 0:
            mask_targets = torch.cat(mask_targets)
        return mask_targets

    def get_target_single(self, pos_assigned_gt_inds, gt_masks):
        device = pos_assigned_gt_inds.device
        num_pos = pos_assigned_gt_inds.size(0)
        if num_pos > 0:
            mask_targets = torch.from_numpy(gt_masks.to_ndarray()).float()
            start = self.out_stride // 2
            stride = self.out_stride
            mask_targets = mask_targets[:, start::stride, start::stride]
            mask_targets = mask_targets.to(device)[pos_assigned_gt_inds]
        else:
            mask_targets = pos_assigned_gt_inds.new_zeros((
                0, gt_masks.height // self.out_stride,
                gt_masks.width // self.out_stride))
        return mask_targets

    def get_seg_masks(self, mask_pred, det_labels, rcnn_test_cfg, ori_shape,
                      scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if not isinstance(mask_pred, torch.Tensor):
            mask_pred = det_labels.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        segms = []
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = det_labels.new_tensor(scale_factor)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        num_chunks = int(
            np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (num_chunks <=
                N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        out_h, out_w = mask_pred.size()[-2:]
        out_h = out_h * self.out_stride
        out_w = out_w * self.out_stride
        im_mask = torch.zeros(
            N,
            out_h,
            out_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        for inds in chunks:
            masks_chunk = _do_paste_mask_hr(
                mask_pred[inds],
                out_h,
                out_w,
                offset=0.)

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[inds] = masks_chunk

        for i in range(N):
            segm = im_mask[i, :img_h, :img_w].cpu().numpy()
            cls_segms[labels[i]].append(segm)
            segms.append(segm)
        return cls_segms, segms

    def get_hr_masks(self, feat, mask_pred, det_bboxes, det_labels,
                     scale_factor):
        """Get high-resolution masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            feat_shape (Tensor): shape (3, )

        Returns:
            list[list]: encoded masks
        """
        if not isinstance(mask_pred, torch.Tensor):
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        level = int(math.log2(self.feat_stride // 4))
        mask_h, mask_w = feat[level].size()[-2:]

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor / self.feat_stride
        rects = bbox_rescale(bboxes, self.rect_scale_factor)

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
                np.ceil(N * mask_h * mask_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'

        im_mask = torch.zeros(
            N,
            1,
            mask_h,
            mask_w,
            device=device,
            dtype=torch.float32)
        im_rect = torch.zeros(
            N, 1, mask_h, mask_w, device=device, dtype=torch.float32)

        mask_pred = mask_pred[range(N), labels][:, None]
        rect_pred = mask_pred.new_ones(mask_pred.size())

        if N == 0:
            return im_mask, im_rect

        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                mask_h,
                mask_w,
                skip_empty=device.type == 'cpu')

            im_mask[(inds, 0) + spatial_inds] = masks_chunk

            rects_chunk, spatial_inds = _do_paste_mask(
                rect_pred[inds], rects[inds], mask_h, mask_w, skip_empty=False)
            im_rect[(inds, 0) + spatial_inds] = rects_chunk
        
        return im_mask, im_rect


def _do_paste_mask_hr(masks, img_h, img_w, offset):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        The whole image will be pasted. It will return a mask of shape
        (N, img_h, img_w) and an empty tuple.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + offset
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + offset
    img_y = img_y / img_h * 2 - 1
    img_x = img_x / img_w * 2 - 1
    img_y = img_y.unsqueeze(dim=0).repeat(N, 1)
    img_x = img_x.unsqueeze(dim=0).repeat(N, 1)
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    return img_masks[:, 0]