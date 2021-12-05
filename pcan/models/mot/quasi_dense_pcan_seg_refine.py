import torch
from mmdet.core import bbox2result
import numpy as np

from pcan.core import segtrack2result
from ..builder import MODELS
from .quasi_dense_pcan_seg import QuasiDenseMaskRCNN

import torch.nn.functional as F

@MODELS.register_module()
class EMQuasiDenseMaskRCNNRefine(QuasiDenseMaskRCNN):

    def fix_modules(self):
        fixed_modules = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_roi_extractor,
            self.roi_head.bbox_head,
            self.roi_head.track_roi_extractor,
            self.roi_head.track_head,
            self.roi_head.mask_roi_extractor,
            self.roi_head.mask_head]
        
        for module in fixed_modules:
            for name, param in module.named_parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _em_iter(self, x, mu):
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)                                 # r * c * n
        for _ in range(self.stage_num):
            z = torch.einsum('rcn,rck->rnk', (x, mu))           # r * n * k
            z = F.softmax(20 * z, dim=2)                        # r * n * k
            z = self._l1norm(z, dim=1)                          # r * n * k
            mu = torch.einsum('rcn,rnk->rck', (x, z))           # r * c * k
            mu = self._l2norm(mu, dim=1)                        # r * c * k
        return mu

    def _prop(self, feat, mu):
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)                             # B * C * N
        z = torch.einsum('bcn,bck->bnk', (x, mu))           # B * N * K
        z = F.softmax(z, dim=2)                             # B * N * K
        return z

    def forward_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        img_metas = img_metas[0]
        frame_id = img_metas[0].get('frame_id', -1)
        x = self.extract_feat(img[0])
        if frame_id == 0:
            self.init_tracker()
            self.memo_banks = [x[0], x[1], x[2]]
            self.mus = [self.mu0, self.mu1, self.mu2]

        x = list(x)
        for i in range(2):
            B, C, H, W = self.memo_banks[i].size()
            protos = self._em_iter(self.memo_banks[i], self.mus[i])
            ref_z = self._prop(x[i], protos)            
            ref_r = torch.einsum('bck,bnk->bcn', (protos, ref_z))
            ref_r = ref_r.view(B, C, H, W)
            self.memo_banks[i] = x[i]
            x[i] = x[i] * 0.75 + ref_r * 0.25
            self.mus[i] = self.mus[i] * 0.5 + protos * 0.5

        x = tuple(x)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = (
            self.roi_head.simple_test(x, img_metas, proposal_list, rescale))
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)
        segm_result, ori_segms, labels_ori  = self.roi_head.get_seg_masks(
            img_metas, det_bboxes, det_labels, det_masks, rescale=rescale)

        update_cls_segms = [[] for _ in range(self.roi_head.bbox_head.num_classes)]
        if track_feats is None:
            from collections import defaultdict
            track_result = defaultdict(list)
            refine_bbox_result = bbox_result
            update_cls_segms = segm_result
        else:
            bboxes, labels, masks, ids, embeds, ref_feats, ref_masks, inds, valids = (
                self.tracker.match(
                    bboxes=det_bboxes,
                    labels=det_labels,
                    masks=det_masks,
                    track_feats=track_feats,
                    frame_id=frame_id))

            mask_preds, mask_feats = masks['mask_pred'], masks['mask_feats']

            refine_segm_result, segms, refine_preds = self.roi_head.simple_test_refine(
                img_metas, mask_feats, mask_preds, bboxes, labels, ref_feats,
                ref_masks, rescale=rescale)

            ori_segms = np.array(ori_segms)
            ori_segms = ori_segms[list(inds.cpu().numpy()), :]
           
            labels_ori = labels_ori[inds]
            valids = list(valids.cpu().numpy()) 
            valids_new = [ind2 for ind2 in range(len(valids)) if valids[ind2] == True]

            ori_segms[valids_new,:] = segms
            ori_segms = list(ori_segms)
            
            for i1 in range(len(ori_segms)):
                update_cls_segms[labels_ori[i1]].append(ori_segms[i1])

            self.tracker.update_memo(ids, bboxes, mask_preds, mask_feats,
                                     refine_preds, embeds, labels, frame_id)

            track_result = segtrack2result(bboxes, labels, segms, ids)

        return dict(bbox_result=bbox_result, segm_result=update_cls_segms,
                    track_result=track_result)
