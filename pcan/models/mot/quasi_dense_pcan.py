import math
import numpy as np
import os
import random

import mmcv
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet.core import bbox2result
from mmcv.runner import auto_fp16, force_fp32
from PIL import ImageColor

from pcan.core import track2result
from ..builder import MODELS, build_tracker
from .quasi_dense import QuasiDenseFasterRCNN
#from .quasi_dense_pcan_seg import QuasiDenseMaskRCNN


@MODELS.register_module()
class EMQuasiDenseFasterRCNN(QuasiDenseFasterRCNN):

    def __init__(self, channels, proto_num, stage_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.proto_num = proto_num
        self.stage_num = stage_num

        for i in range(5):
            protos = torch.Tensor(1, channels, proto_num)
            protos.normal_(0, math.sqrt(2. / proto_num))
            protos = self._l2norm(protos, dim=1)
            self.register_buffer('mu%d' % i, protos)

    @force_fp32(apply_to=('inp', ))
    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @force_fp32(apply_to=('inp', ))
    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_match_indices,
                    ref_img,
                    ref_img_metas,
                    ref_gt_bboxes,
                    ref_gt_labels,
                    ref_gt_match_indices,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    ref_gt_bboxes_ignore=None,
                    ref_gt_masks=None,
                    **kwargs):
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)
        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore, ref_gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()

        x = self.extract_feat(img)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = self.roi_head.simple_test(
            x, img_metas, proposal_list, rescale)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)
        segm_result, _ = self.roi_head.get_seg_masks(
            img_metas, det_bboxes, det_labels, det_masks, rescale=rescale)

        if track_feats is None:
            from collections import defaultdict
            track_result = defaultdict(list)
        else:
            bboxes, labels, masks, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                masks=det_masks,
                track_feats=track_feats,
                frame_id=frame_id)

            _, segms = self.roi_head.get_seg_masks(
                img_metas, bboxes, labels, masks, rescale=rescale)

            track_result = track2result(bboxes, labels, segms, ids)
        return dict(bbox_result=bbox_result, segm_result=segm_result,
                    track_result=track_result)

