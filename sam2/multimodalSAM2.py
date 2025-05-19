import torch
from torch import nn
from sam2.utils.misc_gen import interpolate
from hydra import compose, initialize
import einops
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F
from sam2.model_utils import BackboneOutput, DecoderOutput


class promptableSAM2(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(self, image_size, sam, args):
        super().__init__()
        self.fusion_stages_vis = sam.image_encoder.trunk.stage_ends
        self.image_size = image_size
        self.sam = sam
        pixel_mean= [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def extract_features(self, images):
        B, T = 1, 1
        vis_outs = self.sam.image_encoder.trunk(images)
        orig_size = [tuple(x.shape[-2:]) for x in images]
        backbone_out = self._forward_fpn(vis_outs)
        vision_feats, vision_pos_embeds, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        backbone_output = BackboneOutput(B, T, orig_size, vision_feats, vision_pos_embeds, feat_sizes)
        return backbone_output
        
    def preprocess_visual_features(self, samples, image_size):
        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([self.preprocess(x) for x in samples], dim=0)
        BT = (B, T)
        return samples, BT, orig_size

    def compute_decoder_out_no_mem(self, backbone_out: BackboneOutput, prompt_input, multimask_output):
        current_vision_feats = backbone_out.get_current_feats(0)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
            multimask_output=multimask_output
        )
        decoder_out.compute_mask(self.image_size, backbone_out.orig_size[0])

        return decoder_out.masks, decoder_out.ious, decoder_out.low_res_masks
    
    def forward(self, samples, targets):
        # samples: tensor B*T, C, H, W
        samples, BT, orig_size = self.preprocess_visual_features(samples, self.image_size)

        B, T = BT
        vis_outs = self.sam.image_encoder.trunk(samples)
        backbone_out = self._forward_fpn(vis_outs)
        vision_feats, vision_pos_embeds, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        backbone_output = BackboneOutput(B, T, orig_size, vision_feats, vision_pos_embeds, feat_sizes)
        outputs = []
        for idx in range(B):                        
            #point_coords = positive_coords.unsqueeze(0).float()  # Shape [1, P, 2]
            #point_labels = torch.ones(positive_coords.shape[0], dtype=torch.int32).unsqueeze(0)  # Shape [1, P]
            point_coords = targets['point_coords'][idx]
            point_labels = targets['point_labels'][idx]
            point_inputs = {
                "point_coords": point_coords, # Shape [1, P, 2]
                "point_labels": point_labels  # Shape [1, P]
            }
            decoder_out_w_mem:DecoderOutput = self.compute_decoder_out_no_mem(backbone_output, idx, point_inputs)
            decoder_out_w_mem.compute_mask(self.image_size, backbone_output.orig_size[idx])
                    
            outputs.append({
                "masks": decoder_out_w_mem.masks,
            })

        masks = torch.cat([out["masks"] for out in outputs])
        return {"pred_masks": masks.squeeze(1)}
        

    def _early_fusion_stage(self, samples, T):
        vis = self.sam.image_encoder.trunk.patch_embed(samples)
        vis = vis + self.sam.image_encoder.trunk._get_pos_embed(vis.shape[1:3])

        vis_outs = []
        fusion_stages_vis = [x+1 for x in self.fusion_stages_vis]
        fusion_vis = fusion_stages_vis.copy()
        fusion_vis.insert(0, 0)
        for i, i_v in enumerate(fusion_vis[:-1]):
            vis = self.forw_layer_list(i_v, fusion_vis[i+1], self.sam.image_encoder.trunk.blocks, vis)
            vis_outs.append(vis.permute(0, 3, 1, 2))
        return vis_outs

    def _forward_fpn(self, vis_outs):
        features, pos = self.sam.image_encoder.neck(vis_outs)

        # Discard the lowest resolution features
        features, pos = features[: -1], pos[: -1]
        image_embedding = features[-1]

        backbone_out = {
            "vision_features": image_embedding,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        backbone_out["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        # h, w = x.shape[-2:]
        # padh = image_size - h
        # padw = image_size - w
        x = F.interpolate(x, (self.image_size, self.image_size), align_corners=False, mode='bilinear', antialias=True)
        # x = F.pad(x, (0, padw, 0, padh), value=value)
        return x


def build_sam2(args):
    # image encoder and decoder
    config_file = 'sam2_configs/sam2_hiera_l.yaml'
    sam2_weights = 'pretrain/sam2_hiera_large.pt'
    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=config_file)

        OmegaConf.resolve(cfg)
        sam = instantiate(cfg.model, _recursive_=True)
    state_dict = torch.load(sam2_weights, map_location="cpu")["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = promptableSAM2(image_size=sam.image_size, sam=sam, args=args)
        
            
    return model


