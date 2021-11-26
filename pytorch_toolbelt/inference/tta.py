"""Implementation of GPU-friendly test-time augmentation for image segmentation and classification tasks.

Despite this is called test-time augmentation, these method can be used at training time as well since all
transformation written in PyTorch and respect gradients flow.
"""
from functools import partial
from typing import Tuple, List

from torch import Tensor, nn
from torch.nn.functional import interpolate

from . import functional as F

__all__ = [
    "d4_image2label",
    "d4_image2mask",
    "fivecrop_image2label",
    "tencrop_image2label",
    "fliplr_image2mask",
    "fliplr_image2label",
    "TTAWrapper",
    "MultiscaleTTAWrapper",
]


def fliplr_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    for input image and vertically flipped one.

    :param model:
    :param image:
    :return:
    """
    output = model(image) + model(F.torch_fliplr(image))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def fivecrop_image2label(model: nn.Module, image: Tensor, crop_size: Tuple) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them.

    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(2)), int(image.size(3))
    crop_height, crop_width = crop_size

    assert crop_height <= image_height
    assert crop_width <= image_width

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    assert crop_tl.size(2) == crop_height
    assert crop_tr.size(2) == crop_height
    assert crop_bl.size(2) == crop_height
    assert crop_br.size(2) == crop_height

    assert crop_tl.size(3) == crop_width
    assert crop_tr.size(3) == crop_width
    assert crop_bl.size(3) == crop_width
    assert crop_br.size(3) == crop_width

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2

    crop_cc = image[
        ...,
        center_crop_y : center_crop_y + crop_height,
        center_crop_x : center_crop_x + crop_width,
    ]
    assert crop_cc.size(2) == crop_height
    assert crop_cc.size(3) == crop_width

    output = (
        model(crop_tl)
        + model(crop_tr)
        + model(crop_bl)
        + model(crop_br)
        + model(crop_cc)
    )
    one_over_5 = float(1.0 / 5.0)
    return output * one_over_5


def tencrop_image2label(model: nn.Module, image: Tensor, crop_size: Tuple) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them and from their horisontally-flipped versions (10-Crop TTA).

    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(2)), int(image.size(3))
    crop_height, crop_width = crop_size

    assert crop_height <= image_height
    assert crop_width <= image_width

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    assert crop_tl.size(2) == crop_height
    assert crop_tr.size(2) == crop_height
    assert crop_bl.size(2) == crop_height
    assert crop_br.size(2) == crop_height

    assert crop_tl.size(3) == crop_width
    assert crop_tr.size(3) == crop_width
    assert crop_bl.size(3) == crop_width
    assert crop_br.size(3) == crop_width

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2

    crop_cc = image[
        ...,
        center_crop_y : center_crop_y + crop_height,
        center_crop_x : center_crop_x + crop_width,
    ]
    assert crop_cc.size(2) == crop_height
    assert crop_cc.size(3) == crop_width

    output = (
        model(crop_tl)
        + model(F.torch_fliplr(crop_tl))
        + model(crop_tr)
        + model(F.torch_fliplr(crop_tr))
        + model(crop_bl)
        + model(F.torch_fliplr(crop_bl))
        + model(crop_br)
        + model(F.torch_fliplr(crop_br))
        + model(crop_cc)
        + model(F.torch_fliplr(crop_cc))
    )

    one_over_10 = float(1.0 / 10.0)
    return output * one_over_10


def fliplr_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image) + F.torch_fliplr(model(F.torch_fliplr(image)))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def d4_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug in [F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    image = F.torch_transpose(image)

    for aug in [F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8


def d4_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug, deaug in zip(
        [F.torch_rot90, F.torch_rot180, F.torch_rot270],
        [F.torch_rot270, F.torch_rot180, F.torch_rot90],
    ):
        x = deaug(model(aug(image)))
        output = output + x

    image = F.torch_transpose(image)

    for aug, deaug in zip(
        [F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270],
        [F.torch_none, F.torch_rot270, F.torch_rot180, F.torch_rot90],
    ):
        x = deaug(model(aug(image)))
        output = output + F.torch_transpose(x)

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8


class TTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


class MultiscaleTTAWrapper(nn.Module):
    """
    Multiscale TTA wrapper module
    """

    def __init__(self, model: nn.Module, scale_levels: List[float]):
        """
        Initialize multi-scale TTA wrapper

        :param model: Base model for inference
        :param scale_levels: List of additional scale levels,
            e.g: [0.5, 0.75, 1.25]
        """
        super().__init__()
        assert len(scale_levels)
        self.model = model
        self.scale_levels = scale_levels

    def forward(self, input: Tensor) -> Tensor:
        h = input.size(2)
        w = input.size(3)

        out_size = h, w
        output = self.model(input)

        for scale in self.scale_levels:
            dst_size = int(h * scale), int(w * scale)
            input_scaled = interpolate(
                input, dst_size, mode="bilinear", align_corners=False
            )
            output_scaled = self.model(input_scaled)
            output_scaled = interpolate(
                output_scaled, out_size, mode="bilinear", align_corners=False
            )
            output += output_scaled

        return output / (1 + len(self.scale_levels))
