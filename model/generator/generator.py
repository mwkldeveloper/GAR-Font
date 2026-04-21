import torch
import torch.nn as nn
from functools import partial
import math
from model.modules.blocks import *

from model.generator.gpt import Transformer as GPT
from model.generator.feature_fusion_module import FeatureFusionModule
from model.tokenizer.tokenizer import CNNEncoder



class StyleEncoder(nn.Module):
    def __init__(
        self, 
        C_in, 
        C, 
        C_out, 
        norm='none', 
        activ='relu', 
        pad_type='reflect', 
        sigmoid=False, 
        scale_var=True,
        downsample_ratio=16
    ):
        super().__init__()
        assert downsample_ratio in [8, 16], "downsample_ratio must be 8, 16"
        downsample_times = int(math.log2(downsample_ratio))

        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, scale_var=scale_var)

        layers = []
        layers.append(ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'))

        in_ch = C
        for i in range(downsample_times-1):
            out_ch = in_ch * 2
            
            layers.append(ConvBlk(in_ch, out_ch, 3, 1, 1, downsample=True))
            in_ch = out_ch

        layers.append(ResBlk(in_ch, in_ch, 3, 1))
        layers.append(ResBlk(in_ch, in_ch, 3, 1))
        layers.append(ResBlk(in_ch, 2*in_ch, 3, 1, downsample=True))
        
        layers.append(ResBlk(2*in_ch, C_out))

        self.net = nn.Sequential(*layers)
        self.if_sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.if_sigmoid:
            out = nn.Sigmoid()(out)
        return out
    
class Generator(nn.Module):
    def __init__(
        self,
        content_encoder_args,
        style_encoder_args,
        gpt_args,
        ffm_args,
    ):
        super().__init__()
        self.content_encoder = CNNEncoder(**content_encoder_args)
        self.style_encoder = StyleEncoder(**style_encoder_args)
        self.ffm = FeatureFusionModule(**ffm_args)
        self.gpt = GPT(gpt_args)

    def inference():
        pass
    
    def forward(
        self,
        content_img,                 # [B, C_in, H, W]
        style_imgs,                  # [B, n_ref, C_in, H, W]
        vq_indices = None,           # [B, tar_token_len]
        gpt_valid = None,            # [B]
        gpt_attn_mask = None,        # [B, 1, seq_len, seq_len] 
        aligner_model=None,
        t5_feats=None,
    ):
        B, n_ref, C_in, H, W = style_imgs.shape

        # Content encoding
        encoded_content = self.content_encoder(content_img)         
        
        # Style encoding
        style_imgs_flat = style_imgs.reshape(B * n_ref, C_in, H, W) # [B*n_ref, C_in, h, w]
        style_feats = self.style_encoder(style_imgs_flat) # [B*n_ref, C_out, h, w]
            
        _, C_out, h, w = style_feats.shape
        style_feats = style_feats.view(B, n_ref, C_out, h, w) # [B, n_ref, C_out, h, w]
        if aligner_model:
            style_feats = aligner_model(style_feats, t5_feats) # [B, n_keep+1, C, h, w]
            
        feature_fused = self.ffm(encoded_content, style_feats) # [B, C_out, h, w]
            
        # Fused feature map 
        cat_fused_img_feature_map = torch.cat([encoded_content, feature_fused], dim=1) # [B, 2*C_out, h, w]
        
        # Forward GPT
        logits, gpt_loss = self.gpt(
            idx= vq_indices[:,:-1], 
            imgs_feature_map=cat_fused_img_feature_map,
            targets=vq_indices,
            mask = gpt_attn_mask[:, :, :-1, :-1],
            valid=gpt_valid
        )


        return logits, gpt_loss