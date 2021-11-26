import numpy as np
import torch

import mmcv
from mmdet.core import bbox2roi
from mmdet.models import HEADS

from .quasi_dense_roi_head import QuasiDenseRoIHead


@HEADS.register_module()
class QuasiDenseSegRoIHead(QuasiDenseRoIHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.mask_head is not None

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shaep = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            mask_results = dict(mask_pred=None, mask_feats=None)
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
        return mask_results

    def simple_test(self, x, img_metas, proposal_list, rescale):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        # TODO: support batch inference
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]
        det_masks = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, det_masks, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])

        return det_bboxes, det_labels, det_masks, track_feats

    def get_seg_masks(self, img_metas, det_bboxes, det_labels, det_masks,
                      rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            det_segms = []
            labels = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            segm_result, det_segms, labels = self.mask_head.get_seg_masks(
                det_masks['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result, det_segms, labels
