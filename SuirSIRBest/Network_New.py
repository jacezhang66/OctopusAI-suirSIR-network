import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.models as models
from option import opt
from torchsummary import summary
from thop import profile
from einops import rearrange
import numpy as np
import numbers
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Network_Other import VeilswinTransformerBlock


def IFM_reverse(Jc, A, S, bataDc):
    pred_clear = (Jc - A) / bataDc + S
    return pred_clear


class veil(nn.Module):
    def __init__(self):
        super(veil, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.selfatn_refine = VeilswinTransformerBlock(dim=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_r, x_g, x_b):
        F_x = self.conv(x)
        F_x = self.conv1(F_x)
        x_r = self.sigmoid((1 - x_r) * (x_g + x_b) * 0.5)
        F_x = F_x * x_r
        F_x = self.conv2(F_x)
        F_x = self.selfatn_refine(F_x)

        tensorones_new = F_x

        corresponding_values = self.conv3(tensorones_new)

        return corresponding_values


class FeatureExtractor3(nn.Module):
    def __init__(self):
        super(FeatureExtractor3, self).__init__()
        self.relu = nn.ReLU()

        self.conv_start = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_13 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_14 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_21 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=opt.SW_bias)

        self.conv_22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_21r = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.conv_23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_24 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_31 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=opt.SW_bias)

        self.conv_32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_31r = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.conv_33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_41 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_42 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_41r = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.conv_43 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_44 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_51 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_52 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_51r = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=opt.SW_bias)
        self.conv_53 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.conv_54 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)

        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=opt.SW_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_start(x)

        x_11 = self.relu(self.conv_11(x))
        x_12 = self.conv_12(x_11)
        x_13 = self.relu(x_12 + x)

        x_14 = self.relu(self.conv_13(x_13))
        x_15 = self.conv_14(x_14)
        x_16 = self.relu(x_13 + x_15)

        x_21 = self.relu(self.conv_21(x_16))
        x_22 = self.conv_22(x_21)
        x_16r = torch.max_pool2d(x_16, kernel_size=2, stride=2, padding=0)
        x_16r = self.conv_21r(x_16r)
        x_23 = self.relu(x_22 + x_16r)

        x_24 = self.relu(self.conv_23(x_23))
        x_25 = self.conv_24(x_24)
        x_26 = self.relu(x_25 + x_23)

        x_31 = self.relu(self.conv_31(x_26))
        x_32 = self.conv_32(x_31)
        x_26r = torch.max_pool2d(x_26, kernel_size=2, stride=2, padding=0)
        x_26r = self.conv_31r(x_26r)
        x_33 = self.relu(x_32 + x_26r)

        x_34 = self.relu(self.conv_33(x_33))
        x_35 = self.conv_34(x_34)
        x_36 = self.relu(x_35 + x_33)

        x_41 = F.interpolate(x_36, scale_factor=2, mode="nearest")
        x_42 = self.relu(self.conv_41(x_41))
        x_43 = self.conv_42(x_42)
        x_41r = self.conv_41r(x_41)
        x_44 = self.relu(x_43 + x_41r)

        x_45 = self.relu(self.conv_43(x_44))
        x_46 = self.conv_44(x_45)
        x_47 = self.relu(x_46 + x_44)

        x_51 = F.interpolate(x_47, scale_factor=2, mode="nearest")
        x_52 = self.relu(self.conv_51(x_51))
        x_53 = self.conv_52(x_52)
        x_51r = self.conv_51r(x_51)
        x_54 = self.relu(x_53 + x_51r)

        x_55 = self.relu(self.conv_53(x_54))
        x_56 = self.conv_54(x_55)
        x_57 = self.relu(x_56 + x_54)

        x_final1 = self.conv_final(x_57)
        x_final1 = self.sigmoid(x_final1)

        return x_final1


class PixPromptGB(nn.Module):
    def __init__(self, ppix_dim, ppix_len, ppix_size, ppixlin_dim):
        super(PixPromptGB, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, ppix_len, ppix_dim, ppix_size, ppix_size))
        self.linear_layer = nn.Linear(ppixlin_dim, ppix_len)
        self.conv3x3 = nn.Conv2d(ppix_dim, ppix_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        prompt_weights = F.softmax(self.linear_layer(x.mean(dim=(-2, -1))), dim=1)
        prompt1 = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        prompt2 = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt1 * prompt2, dim=1)

        return self.conv3x3(prompt)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.prompt = PixPromptGB(ppix_dim=self.dim, ppix_len=5, ppix_size=input_resolution[0], ppixlin_dim=self.dim)
        self.coonv1 = nn.Conv2d(in_channels=120, out_channels=60, kernel_size=3, padding=1)

    def calculate_mask(self, x_size):

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        step2 = x

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x_prompt_out = self.prompt(step2.permute(0, 3, 1, 2))

        x_step1_out = x.permute(0, 3, 1, 2)

        x_cat_out = torch.cat([x_prompt_out, x_step1_out], dim=1)

        x_out = self.coonv1(x_cat_out)

        x_out_p = x_out.permute(0, 1, 2, 3)

        x = x_out_p.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                      norm_layer=None)

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                          norm_layer=None)

    def forward(self, x, x_size):
        x_step1 = self.residual_group(x, x_size)
        x_step2 = self.patch_unembed(x_step1, x_size)
        x_step3 = self.conv(x_step2)
        x_stepout = self.patch_embed(x_step3) + x

        return x_stepout

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class S_estimator(nn.Module):

    def __init__(self, img_size=256, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[5, 5, 5, 5], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):

        super(S_estimator, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size)

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean)

        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = x + self.conv_last(res)

        x = x + self.mean

        return x[:, :, :H, :W]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops


class suirSIR(nn.Module):
    def __init__(self):
        super(suirSIR, self).__init__()
        self.smodel = S_estimator(img_size=(256, 256), window_size=8, depths=[4, 4, 4, 4],
                                  embed_dim=60, num_heads=[5, 5, 5, 5], mlp_ratio=2)

        self.weightsharedEnc2 = FeatureExtractor3()
        self.weightsharedEnc4 = FeatureExtractor3()
        self.veil = veil()

        self.conv1 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        s = self.smodel(x)
        t = self.weightsharedEnc2(x)

        x_r = x[:, 0:1, :, :]
        x_g = x[:, 1:2, :, :]
        x_b = x[:, 2:3, :, :]

        A = self.veil(x=x, x_r=x_r, x_g=x_g, x_b=x_b)

        pred_img = IFM_reverse(Jc=x, A=A, S=s, bataDc=t)

        new_t = self.weightsharedEnc4(pred_img)

        regraded_img = (pred_img - s) * new_t + A

        betaD1 = t
        betaD2 = new_t

        return pred_img, regraded_img, betaD1, betaD2, s, A


if __name__ == '__main__':
    model = suirSIR()
    model.cuda()  # your model

    # thop计算网络参数
    input = torch.randn(1, 3, 256, 256)
    input = input.cuda()

    min_value = input.min()
    max_value = input.max()
    input = (input - min_value) / (max_value - min_value)

    flops, params = profile(model, inputs=(input,))

    print(f'FLOPs: {flops / 1e9} G, Params: {params / 1e6} M')

    # 打印张量尺寸

    x, y, a, b = model(input)
    print(f"张量尺寸{x.size()}")
    print(f"张量尺寸{y.size()}")
    print(f"张量尺寸{a.size()}")
    print(f"张量尺寸{b.size()}")
