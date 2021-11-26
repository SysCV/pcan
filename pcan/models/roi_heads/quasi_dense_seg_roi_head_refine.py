import numpy as np
import torch

import mmcv
from mmdet.core import bbox2roi
from mmdet.models import HEADS, build_head

from .quasi_dense_seg_roi_head import QuasiDenseSegRoIHead


@HEADS.register_module()
class QuasiDenseSegRoIHeadRefine(QuasiDenseSegRoIHead):

    def __init__(self, refine_head=None, double_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert refine_head is not None
        self.init_refine_head(refine_head)
        self.double_train = double_train

    @property
    def with_refine(self):
        return hasattr(self, 'refine_head') and self.refine_head is not None

    def init_refine_head(self, refine_head):
        self.refine_head = build_head(refine_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        self.refine_head.init_weights()

    def _refine_forward(self, key_feats, key_masks, key_labels, ref_feats,
                        ref_masks):
        num_rois = key_masks.size(0)
        inds = torch.arange(0, num_rois, device=key_masks.device).long()
        key_masks = key_masks[inds, key_labels].unsqueeze(dim=1)
        ref_masks = ref_masks[inds, key_labels].unsqueeze(dim=1)

        if self.double_train and self.training:
            ref_masks = self.refine_head(ref_feats, ref_masks, ref_feats,
                                         ref_masks).detach()
        #print('ref feats:', ref_feats.shape)
        #print('key feats:', key_feats.shape)
        refine_pred = self.refine_head(ref_feats, ref_masks, key_feats,
                                       key_masks)
        refine_results = dict(refine_pred=refine_pred)
        return refine_results

    def _refine_forward_train(self, key_sampling_results, ref_sampling_results,
                              key_mask_results, ref_mask_results, x, ref_x,
                              gt_match_inds):
        num_key_rois = [len(res.pos_bboxes) for res in key_sampling_results]
        key_pos_pids = [
            gt_match_ind[res.pos_assigned_gt_inds]
            for res, gt_match_ind in zip(key_sampling_results, gt_match_inds)]
        key_pos_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_embeds = torch.split(
            self._track_forward(x, key_pos_bboxes), num_key_rois)

        num_ref_rois = [len(res.pos_bboxes) for res in ref_sampling_results]
        ref_pos_pids = [
            res.pos_assigned_gt_inds for res in ref_sampling_results]
        ref_pos_bboxes = [res.pos_bboxes for res in ref_sampling_results]
        ref_embeds = torch.split(
            self._track_forward(ref_x, ref_pos_bboxes), num_ref_rois)

        valids, ref_inds = self.refine_head.match(
            key_embeds, ref_embeds, key_pos_pids, ref_pos_pids)

        def valid_select(inputs, num_splits, inds):
            inputs = torch.split(inputs, num_splits)
            inputs = torch.cat(
                [input_[ind] for input_, ind in zip(inputs, inds)])
            return inputs

        key_feats = valid_select(
            key_mask_results['mask_feats'], num_key_rois, valids)
        key_masks = valid_select(
            key_mask_results['mask_pred'], num_key_rois, valids)
        key_targets = valid_select(
            key_mask_results['mask_targets'], num_key_rois, valids)
        key_labels = torch.cat(
            [res.pos_gt_labels[valid]
            for res, valid in zip(key_sampling_results, valids)])
        ref_feats = valid_select(
            ref_mask_results['mask_feats'], num_ref_rois, ref_inds)
        ref_masks = valid_select(
            ref_mask_results['mask_pred'], num_ref_rois, ref_inds)

        if key_masks.size(0) == 0:
            key_feats = key_mask_results['mask_feats']
            key_masks = key_mask_results['mask_pred']
            key_targets = key_mask_results['mask_targets']
            key_labels = torch.cat([
                res.pos_gt_labels for res in key_sampling_results])
            ref_feats = key_feats.detach()
            ref_masks = key_masks.detach()

        refine_results = self._refine_forward(key_feats, key_masks, key_labels,
                                              ref_feats, ref_masks)
        refine_targets = key_targets
        loss_refine = self.refine_head.loss_mask(
            refine_results['refine_pred'].squeeze(dim=1), refine_targets)
        refine_results.update(loss_refine=loss_refine,
                              refine_targets=refine_targets)
        return refine_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices, # defines the gt box at current image matching relation with the boxes in the reference image, unmatch given -1
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      *args,
                      **kwargs):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
    
        losses = dict()
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results = []
        for i in range(num_imgs):
            key_assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            key_sampling_result = self.bbox_sampler.sample(
                key_assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(key_sampling_result)

        key_mask_results = self._mask_forward_train(
            x, key_sampling_results, None, gt_masks, img_metas)

        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        ref_sampling_results = []
        for i in range(num_imgs):
            ref_assign_result = self.bbox_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        ref_mask_results = self._mask_forward_train(
            ref_x, ref_sampling_results, None, ref_gt_masks, ref_img_metas)

        refine_results = self._refine_forward_train(
            key_sampling_results, ref_sampling_results, key_mask_results,
            ref_mask_results, x, ref_x, gt_match_indices)
        if refine_results['loss_refine'] is not None:
            losses.update(loss_refine=refine_results['loss_refine'])

        return losses

    def simple_test_refine(self, img_metas, key_feats, key_masks, key_bboxes,
                           key_labels, ref_feats, ref_masks, rescale=False):
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if key_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            det_segms = []
            refine_preds = key_masks.new_full(key_masks.size())
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    key_bboxes.device)
            _bboxes = (
                key_bboxes[:, :4] * scale_factor if rescale else key_bboxes)
            refine_results = self._refine_forward(
                key_feats, key_masks, key_labels, ref_feats, ref_masks)
            
            key_masks_cls = key_masks[range(len(key_masks)), key_labels,:, :].unsqueeze(1)
            select_inds = torch.where(key_labels >= 10)
            refine_preds = refine_results['refine_pred']
            refine_preds[select_inds] = key_masks_cls[select_inds]
           
            segm_result, det_segms = self.refine_head.get_seg_masks(
                refine_preds, _bboxes, key_labels, self.test_cfg, ori_shape,
                scale_factor, rescale)
        return segm_result, det_segms, refine_preds
