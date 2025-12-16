# coding=utf-8
#
# Code mainly copied from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py
# and adjusted for Jina CLIP

from typing import Tuple, Union

import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.models.clip import CLIPProcessor

from .transform import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, image_transform

""" Jina CLIP processor implementation """


class JinaCLIPProcessor(CLIPProcessor):
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'


""" Jina CLIP image processor implementation """


class JinaCLIPImageProcessor(BaseImageProcessor):
    model_input_names = ['pixel_values']
    _valid_processor_keys = [
        'size',
        'mean',
        'std',
        'resize_mode',
        'interpolation',
        'fill_color',
    ]

    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 224,
        mean: Union[float, Tuple[float]] = OPENAI_DATASET_MEAN,
        std: Union[float, Tuple[float]] = OPENAI_DATASET_STD,
        resize_mode: str = 'shortest',
        interpolation: str = 'bicubic',
        fill_color: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.mean = mean
        self.std = std
        self.resize_mode = resize_mode
        self.interpolation = interpolation
        self.fill_color = fill_color
        self.transform = self._build_transform()

    def _build_transform(self):
        return image_transform(
            image_size=self.size,
            is_train=False,
            mean=self.mean,
            std=self.std,
            resize_mode=self.resize_mode,
            interpolation=self.interpolation,
            fill_color=self.fill_color,
            aug_cfg=None,
        )

    def to_dict(self):
        output = super().to_dict()
        output.pop('transform')
        return output

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        _transform_needs_rebuild = False
        for k, v in kwargs.items():
            if k in self._valid_processor_keys:
                if v != getattr(self, k):
                    setattr(self, k, v)
                    _transform_needs_rebuild = True

        if _transform_needs_rebuild:
            self.transform = self._build_transform()

        images = make_list_of_images(images)
        out = torch.stack([self.transform(image) for image in images], dim=0)
        return BatchFeature(data={'pixel_values': out})
