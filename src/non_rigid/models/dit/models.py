# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import omegaconf
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Optional

from non_rigid.nets.dgcnn import DGCNN
from non_rigid.models.dit.relative_encoding import RotaryPositionEncoding3D, MultiheadRelativeAttentionWrapper

torch.set_printoptions(precision=8, sci_mode=True)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Custom Attention Layers                       #
#################################################################################


class CrossAttention(nn.Module):
    """
    Cross attention layer adapted from
    https://github.com/pprp/timm/blob/e9aac412de82310e6905992e802b1ee4dc52b5d1/timm/models/crossvit.py#L132
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim_x % num_heads == 0, "dim x must be divisible by num_heads"
        head_dim = dim_x // num_heads
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.wk = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.wv = nn.Linear(dim_y, dim_x, bias=qkv_bias)
        self.proj = nn.Linear(dim_x, dim_x)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, Cx = x.shape
        # _, _, Cy = y.shape
        _, Ny, Cy = y.shape
        q = (
            self.wq(x)
            .reshape(B, N, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.wk(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.wv(y)
            .reshape(B, Ny, self.num_heads, Cx // self.num_heads)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core DiT Layers                               #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTRelativeBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output

        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x)
        return x

class DiTCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim_x=hidden_size,
            dim_y=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_mca, scale_mca), y
        )
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        return x


class DiTRelativeCrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning 
    and 3D relative self attention + 3D relative scene cross attention.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadRelativeAttentionWrapper(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            bias=True
        )
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, x_pos=None, y_pos=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_x,
            scale_x,
            gate_x,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            query=x, key=x, value=x, rotary_pe=(x_pos, x_pos)
        )[0] # [0] is the attention output
        
        x = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            query=x, key=y, value=y, rotary_pe=(x_pos, y_pos)
        )[0] # [0] is the attention output
        
        x = x + gate_x.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_x, scale_x)
        )
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#################################################################################
#                                 Core DiT Models                               #
#################################################################################

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class LinearRegressionModel(nn.Module):
    """
    Linear regression baseline, with attention - no diffusion.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            model_cfg=None,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channels = 3

        # initializing embedder for action point cloud
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # initializing embedder for anchor point cloud
        self.y_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        class LinearCrossBlock(nn.Module):
            """
            Cross attention block for linear regression model.
            """
            def __init__(self, hidden_size, num_heads, mlp_ratio):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.cross_attn = CrossAttention(
                    dim_x=hidden_size,
                    dim_y=hidden_size,
                    num_heads=num_heads,
                    qkv_bias=True,
                )
                mlp_hidden_dim = int(hidden_size * mlp_ratio)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                )

            def forward(self, x, y):
                x = self.cross_attn(self.norm(x), y)
                x = self.mlp(x)
                return x
            
        class LinearFinalLayer(nn.Module):
            """
            Final layer of the linear regression model.
            """
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.linear = nn.Linear(hidden_size, out_channels, bias=True)

            def forward(self, x):
                x = self.norm_final(x)
                x = self.linear(x)
                return x


        # TODO: DUPLICATE THE CROSS ATTENTION LAYER BASED ON DEPTH
        self.blocks = nn.ModuleList(
            [
                LinearCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = LinearFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of linear regression model.
        """

        if self.model_cfg.center_noise:
            raise NotImplementedError("Center noise not implemented for linear regression model.")
        if self.model_cfg.rotary:
            raise NotImplementedError("Rotary not implemented for linear regression model.")
        
        # encode x and y
        x = torch.transpose(self.x_embedder(x), -1, -2)
        y = torch.transpose(self.y_embedder(y), -1, -2)
        # forward pass through cross attention blocks
        for block in self.blocks:
            x = block(x, y)
        
        # final layer
        x = self.final_layer(x)
        x = torch.transpose(x, -1, -2)
        return x


### CREATING NEW DiT (unconditional) FOR POINT CLOUD INPUTS ###
class DiT_PointCloud_Unc(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        # x_embedder is conv1d layer instead of 2d patch embedder
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=1, stride=1, padding=0, bias=True)
        # no pos_embed, or y_embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, L, 3) tensor of spatial inputs (point clouds)
        t: (N,) tensor of diffusion timesteps
        """
        # concat x and pos
        x = torch.cat((x, x0), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = torch.transpose(x, -1, -2)
        return x


class DiT_PointCloud(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses scene-level self-attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        # Encoder for current timestep x features
        self.x_embedder = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeBlock if self.model_cfg.rotary else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, L, 3) tensor of spatial inputs (point clouds)
        t: (N,) tensor of diffusion timesteps
        x0: (N, L, 3) tensor of un-noised x (e.g. scene) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            x0 = x0 - relative_center

        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))

        # encode x, x0 features
        x = torch.cat((x, x0), dim=1)
        x = torch.transpose(self.x_embedder(x), -1, -2)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, t_emb, x_pos)
            else:
                x = block(x, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = torch.transpose(x, -1, -2)
        return x
    

class DiT_PointCloud_Cross(nn.Module):
    """
    Diffusion Transformer adapted for point cloud inputs. Uses object-centric cross attention.
    """
    def __init__(
            self,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        # Rotary embeddings for relative positional encoding
        if self.model_cfg.rotary:
            self.rotary_pos_enc = RotaryPositionEncoding3D(hidden_size)
        else:
            self.rotary_pos_enc = None

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")
        
        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        block_fn = DiTRelativeCrossBlock if self.model_cfg.rotary else DiTCrossBlock
        self.blocks = nn.ModuleList(
            [
                block_fn(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        # noise-centering, if enabled
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        # rotary position embedding, if enabled
        if self.model_cfg.rotary:
            x_pos = self.rotary_pos_enc(x.permute(0, 2, 1))
            y_pos = self.rotary_pos_enc(y.permute(0, 2, 1))

        # encode x, y, x0 features
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 features must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat([x_emb, x0_emb], dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        # timestep embedding
        t_emb = self.t_embedder(t)

        # forward pass through DiT blocks
        for block in self.blocks:
            if self.model_cfg.rotary:
                x = block(x, y_emb, t_emb, x_pos, y_pos)
            else:
                x = block(x, y_emb, t_emb)

        # final layer
        x = self.final_layer(x, t_emb)
        x = x.permute(0, 2, 1)
        return x





class DiT_PointCloud_Unc_Cross(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional, with scene cross attention
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """
        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center
        
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat((x_emb, x0_emb), dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)

        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, y_emb, c)

        x = self.final_layer(x, c)

        x = x.permute(0, 2, 1)

        return x

class Rel3D_DiT_PointCloud_Unc_Cross(nn.Module):
    """
    Diffusion model with a Transformer backbone - point cloud, unconditional, with scene cross attention, and relative 3D positional encoding and attention.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_cfg=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = 6 if learn_sigma else 3
        self.num_heads = num_heads
        self.model_cfg = model_cfg

        self.relative_3d_encoding = RotaryPositionEncoding3D(hidden_size)

        x_encoder_hidden_dims = hidden_size
        if self.model_cfg.x_encoder is not None and self.model_cfg.x0_encoder is not None:
            # We are concatenating x and x0 features so we halve the hidden size
            x_encoder_hidden_dims = hidden_size // 2

        # Encoder for current timestep x features       
        if self.model_cfg.x_encoder == "mlp":
            # x_embedder is conv1d layer instead of 2d patch embedder
            self.x_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        else:
            raise ValueError(f"Invalid x_encoder: {self.model_cfg.x_encoder}")
        
        # Encoder for y features
        if self.model_cfg.y_encoder == "mlp":
            self.y_embedder = nn.Conv1d(
                in_channels,
                hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.y_encoder == "dgcnn":
            self.y_embedder = DGCNN(
                input_dims=in_channels, emb_dims=hidden_size
            )
        else:
            raise ValueError(f"Invalid y_encoder: {self.model_cfg.y_encoder}")            

        # Encoder for x0 features
        if self.model_cfg.x0_encoder == "mlp":
            self.x0_embedder = nn.Conv1d(
                in_channels,
                x_encoder_hidden_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        elif self.model_cfg.x0_encoder == "dgcnn":
            self.x0_embedder = DGCNN(
                input_dims=in_channels, emb_dims=x_encoder_hidden_dims
            )
        elif self.model_cfg.x0_encoder is None:
            pass
        else:
            raise ValueError(f"Invalid x0_encoder: {self.model_cfg.x0_encoder}")

        # Timestamp embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTRelativeCrossBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        
        # functionally setting patch size to 1 for a point cloud
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize x_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT with scene cross attention.

        Args:
            x (torch.Tensor): (B, D, N) tensor of batched current timestep x (e.g. noised action) features
            t (torch.Tensor): (B,) tensor of diffusion timesteps
            y (torch.Tensor): (B, D, N) tensor of un-noised scene (e.g. anchor) features
            x0 (Optional[torch.Tensor]): (B, D, N) tensor of un-noised x (e.g. action) features
        """

        if self.model_cfg.center_noise:
            relative_center = torch.mean(x, dim=2, keepdim=True)
            x = x - relative_center
            y = y - relative_center

        # Get x and y relative 3D positional encoding
        x_pos = self.relative_3d_encoding(x.permute(0, 2, 1))
        y_pos = self.relative_3d_encoding(y.permute(0, 2, 1))
        
        # Get x and y features        
        x_emb = self.x_embedder(x)

        if self.model_cfg.x0_encoder is not None:
            assert x0 is not None, "x0 must be provided if x0_encoder is not None"
            x0_emb = self.x0_embedder(x0)
            x_emb = torch.cat((x_emb, x0_emb), dim=1)

        if self.model_cfg.y_encoder is not None:
            y_emb = self.y_embedder(y)
            y_emb = y_emb.permute(0, 2, 1)

        x = x_emb.permute(0, 2, 1)
        
        c = self.t_embedder(t)

        for i, block in enumerate(self.blocks):
            x = block(x, y_emb, c, x_pos, y_pos)

        x = self.final_layer(x, c)

        x = x.permute(0, 2, 1)
            
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
