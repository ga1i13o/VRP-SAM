from dataclasses import dataclass, field
from torch import Tensor
import torch

from sam2.modeling.sam2_utils import postprocess_masks
# from tools.metrics import db_eval_boundary, db_eval_iou
from sam2.utils.misc_gen import NestedTensor, nested_tensor_from_tensor_list, interpolate


class DDPWrapper:
    def __init__(self, ddp_module):
        self.ddp_module = ddp_module

    def __call__(self, *args, **kwargs):
        return self.ddp_module(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.ddp_module, name):
            attr = getattr(self.ddp_module, name)

            #  Make sure to return DDPWrapper instead of directly returning the attribute in case of a callable such as state.model.train()
            if callable(attr):

                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if result is self.ddp_module:
                        return self
                    return result

                return wrapper
            return attr

        if hasattr(self.ddp_module.module, name):
            return getattr(self.ddp_module.module, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

@dataclass
class DecoderOutput:
    """Class for keeping decoder outputs."""
    ious: Tensor = None
    low_res_masks: Tensor = None
    high_res_masks: Tensor = None
    obj_ptr: Tensor = None
    object_score_logits: Tensor = None
    hyper_in: Tensor = None
    object_score: Tensor = None
    pix_feat_with_mem: Tensor = None
    masks: Tensor = None

    def compute_mask(self, image_size, original_size):
        assert self.low_res_masks is not None, 'before calling compute_mask you have to set \'low_res_masks\''
        self.masks = postprocess_masks(
            self.low_res_masks,
            image_size,
            input_size=original_size,
            original_size=original_size,
        )

    def move_to_cpu(self):
        self.low_res_masks = self.low_res_masks.cpu()
        self.high_res_masks =  self.high_res_masks.cpu()
        self.obj_ptr = self.obj_ptr.cpu()
        self.object_score_logits = self.object_score_logits.cpu()
        self.hyper_in = self.hyper_in.cpu()
        self.object_score = self.object_score.cpu()
        self.masks = self.masks.cpu()
        self.pix_feat_with_mem = self.pix_feat_with_mem.cpu()
        return self


@dataclass
class BackboneOutput:
    """Class for keeping backbone outputs."""
    B: int = None
    T: int = None
    orig_size: list = None
    vision_feats: list = None
    vision_pos_embeds: list = None
    feat_sizes: list = None

    def get_current_feats(self, idx):
        current_vision_feats = [x[:, idx:idx + 1, :] for x in self.vision_feats]
        return current_vision_feats

    def get_current_pos_embeds(self, idx):
        current_vision_pos_embeds = [x[:, idx:idx + 1, :] for x in self.vision_pos_embeds]
        return current_vision_pos_embeds
    
    def get_high_res_features(self, current_vision_feats):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], self.feat_sizes[:-1])
        ]
        return high_res_features

    def move_to_cpu(self):
        self.vision_feats = [x.cpu() for x in self.vision_feats]
        self.vision_pos_embeds = [x.cpu() for x in self.vision_pos_embeds]
        return self

