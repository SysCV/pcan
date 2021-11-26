import numpy as np
import pycocotools.mask as mask_util


# TODO: move this function to more proper place
def encode_track_results(track_results):
    """Encode bitmap mask to RLE code.

    Args:
        track_results (list | tuple[list]): track results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    for id, roi in track_results.items():
        roi['segm'] = mask_util.encode(
            np.array(roi['segm'][:, :, np.newaxis], order='F',
                     dtype='uint8'))[0]  # encoded with RLE
    return track_results