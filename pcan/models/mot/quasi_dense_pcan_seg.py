from mmdet.core import bbox2result

from pcan.core import segtrack2result
from ..builder import MODELS
from .quasi_dense import QuasiDenseFasterRCNN
from .quasi_dense import random_color
from .quasi_dense_pcan import EMQuasiDenseFasterRCNN


import mmcv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

@MODELS.register_module()
class QuasiDenseMaskRCNN(EMQuasiDenseFasterRCNN):

    def __init__(self, fixed=False, *args, **kwargs):
        super().__init__(channels = 256, proto_num = 30, stage_num=3 ,*args, **kwargs)
        if fixed:
            self.fix_modules()

    def fix_modules(self):
        fixed_modules = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_roi_extractor,
            self.roi_head.bbox_head,
            self.roi_head.track_roi_extractor,
            self.roi_head.track_head]
        for module in fixed_modules:
            # print('fixed ======================')
            for name, param in module.named_parameters():
                param.requires_grad = False

    def forward_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        img_metas = img_metas[0]
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()

        x = self.extract_feat(img[0])
        # ref_x = self.extract_feat(ref_img)
        #x = self.em(x, ref_x)

        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = (
            self.roi_head.simple_test(x, img_metas, proposal_list, rescale))
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)
        segm_result, _, _ = self.roi_head.get_seg_masks(
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

            _, segms, _ = self.roi_head.get_seg_masks(
                img_metas, bboxes, labels, masks, rescale=rescale)

            track_result = segtrack2result(bboxes, labels, segms, ids)
        return dict(bbox_result=bbox_result, segm_result=segm_result,
                    track_result=track_result)

    def show_result(self,
                    img,
                    result,
                    show=False,
                    out_file=None,
                    score_thr=0.3,
                    draw_track=True):
        track_result = result['track_result']
        img = mmcv.bgr2rgb(img)

        img = mmcv.imread(img)

        for id, item in track_result.items():
            bbox = item['bbox']
            if bbox[-1] <= score_thr:
                continue
            color = (np.array(random_color(id)) * 256).astype(np.uint8)
            mask = item['segm']
            img[mask] = img[mask] * 0.5 + color * 0.5

        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.autoscale(False)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        for id, item in track_result.items():
            bbox = item['bbox']
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            w = bbox_int[2] - bbox_int[0] + 1
            h = bbox_int[3] - bbox_int[1] + 1
            color = random_color(id)
            plt.gca().add_patch(
                Rectangle(left_top, w, h, edgecolor=color, facecolor='none'))
            label_text = '{}'.format(int(id))
            bg_height = 12
            bg_width = 10
            bg_width = len(label_text) * bg_width
            plt.gca().add_patch(
                Rectangle((left_top[0], left_top[1] - bg_height),
                        bg_width,
                        bg_height,
                        edgecolor=color,
                        facecolor=color))
            plt.text(left_top[0] - 1, left_top[1], label_text, fontsize=5)

        if out_file is not None:
            mmcv.imwrite(img, out_file)
            plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.clf()
        return img
