import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=3, ratio=0.5):
        super().__init__()

        self.window_size = window_size
        self.ratio = ratio
        cdim = dim + k
        embed_dim = window_size**2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            # LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        # offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))
            num_keep_node = int(N * r)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, shift_size=0, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        self.ratio = ratio
        k = 5
        d = 3

        self.shift_size = shift_size

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)

        # Conv
        # self.conv_sptial = nn.Conv2d(dim, dim, kernel_size=3, bias=True, groups=dim, padding=1)
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d)
        )

        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size,ratio=ratio)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        v = self.project_v(shifted_x)

        if True:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if self.shift_size > 0:
                condition_wind = torch.roll(condition_wind, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode)

        # cyclic shift
        x_warped = flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')

        if self.shift_size > 0:
            shifted_x_warped = torch.roll(x_warped, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x_warped = x_warped


        q = shifted_x_warped
        k = shifted_x_warped
        qk = torch.cat([q,k],dim=1)

        # Attn branch
        vs = v*sa

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)
            qk1 = qk*mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)


        #calculate attention: Softmax(Q@K)@V
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)',
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            attn_out = attn_out

        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)

        if self.training:
            return out, torch.mean(mask,dim=1)
        return out

class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=True, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=5, stride=1, padding=2, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Block(nn.Module):
    def __init__(self, n_feats, window_size=8, shift_size=0, ratio=0.5):
        super(Block,self).__init__()

        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats,window_size=window_size,shift_size=shift_size,ratio=ratio)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)

    def forward(self,x,condition_global=None):
        if self.training:
            res, decision = self.mixer(x,condition_global)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            return x, decision
        else:
            res = self.mixer(x,condition_global)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            return x

class Group(nn.Module):
    def __init__(self, n_feats, n_block, window_size=8, ratio=0.5):
        super(Group, self).__init__()

        self.n_feats = n_feats

        self.body = nn.ModuleList([Block(n_feats, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, ratio=ratio) for i in range(n_block)])

        self.offset_conv = nn.Sequential(nn.Conv2d(1, n_feats, 1, 1, 0, bias=True),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv2d(n_feats, n_feats, 1, 1, 0, bias=True),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body_tail = DCNv2Pack(n_feats, n_feats, 3, padding=1)

    def forward(self,x,condition_global=None):
        decision = []
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.append(mask)
            x = self.body_tail(x,self.offset_conv(condition_global)) + shortcut
            return x, decision
        else:
            for _, blk in enumerate(self.body):
                x = blk(x,condition_global)
            x = self.body_tail(x,self.offset_conv(condition_global)) + shortcut
            return x

@ARCH_REGISTRY.register()
class CAMixerOSR(nn.Module):
    def __init__(self, n_block=[6,6,6,6], n_group=4, n_colors=3, n_feats=60, scale=4, ratio=0.5, tile=None):
        super().__init__()

        if n_colors == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range =1.0
        self.window_sizes = 16
        self.tile = tile
        self.scale = scale

        # define head module
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.norm = LayerNorm(n_feats)
        self.body = nn.ModuleList([Group(n_feats, n_block=n_block[i], window_size=self.window_sizes, ratio=ratio) for i in range(n_group)])

        self.offset_conv = nn.Sequential(nn.Conv2d(1, n_feats, 1, 1, 0, bias=True),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv2d(n_feats, n_feats, 1, 1, 0, bias=True),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.dcn = DCNv2Pack(n_feats, n_feats, 3, padding=1)
        # define tail module

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward_origin(self, x, condition):
        decision = []
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.head(x)

        condition_global = condition
        shortcut = x.clone()

        x = self.norm(x)
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.extend(mask)
        else:
            for _, blk in enumerate(self.body):
                x = blk(x,condition_global)

        x = self.body_tail(x)
        x = self.dcn(x, self.offset_conv(condition_global))
        x = x + shortcut
        x = self.tail(x)
        x = x / self.img_range + self.mean

        if self.training:
            return x[:, :, 0:H*self.scale, 0:W*self.scale], (torch.mean(torch.cat(decision,dim=1),dim=(0,1))-0.5)**2 #+ torch.var(torch.cat(decision,dim=1),dim=(0,1))
        else:
            return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def forward(self, img_lq, condition=None, tile=None):
        tile = self.tile
        if tile is None or self.training:
            # for test
            if condition is None: 
                condition = torch.zeros(1,1,img_lq.shape[-2],img_lq.shape[-1])
            # test the image as a whole
            output = self.forward_origin(img_lq,condition)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            tile_overlap = 8
            sf = self.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.forward_origin(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output

    def check_image_size(self, x, ):
        _, _, h, w = x.size()
        wsize = self.window_sizes
        # for i in range(1, len(self.window_sizes)):
        #     wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


if __name__ == '__main__':

    x = torch.randn((1,3,64,64))
    p = torch.randn((1,1,64,64))
    net = CAMixerOSR(scale=4)
    net.eval()
    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))
    print(net(x,p).shape)




        