import torch
import torch.nn as nn
from .basic_modules import A_CDP, At_CDP
import sys, os
from .net import Block, SELayer, LayerNorm, HERB, GRN, DCFB
from typing import Union
from timm.layers import DropPath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from diffusers import UNet2DModel
from diffusers.configuration_utils import register_to_config


class TDB(UNet2DModel):
    @register_to_config
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time_embed_dim = self.config.time_embedding_dim or self.config.block_out_channels[0] * 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.config.in_channels, self.config.block_out_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(self.config.block_out_channels[0], self.config.block_out_channels[0], kernel_size=3, padding=1),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.config.block_out_channels[0], self.config.block_out_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(self.config.block_out_channels[0], self.config.out_channels, kernel_size=3, padding=1),
        ) 
        

    def forward(self, 
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
                ):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=torch.float32)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        
        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample



class Head(nn.Module):
    def __init__(
            self,
            in_channels=1,
            dim = 32,
            drop_path_rate=0
    ):
        super().__init__()
        self.dim = dim
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, kernel_size=3, padding=1),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim),
            *[Block(dim=self.dim, drop_path=dp_rates[j]) for j in range(1)],
        )

    def forward(self, x):
        x = self.head(x)
        return x

class Tail(nn.Module):
    def __init__(
            self,
            dim=32,
            out_channels=1,
            drop_path_rate=0
    ):
        super().__init__()
        self.dim = dim
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]
        self.tail = nn.Sequential(
            *[Block(dim=self.dim, drop_path=dp_rates[j]) for j in range(1)],
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim),
            nn.Conv2d(self.dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.tail(x)
        return x

class RFMB(nn.Module):
    def __init__(
            self,
            embed_dim=32,
    ):
        super(RFMB, self).__init__()
        self.dim = embed_dim
        self.attention = nn.Sequential(Block(self.dim, drop_path=0),
                                       nn.ReLU(inplace=True),
                                       HERB(self.dim, kernel_size=3, reduction=16),
                                       Block(self.dim, drop_path=0),
                                       HERB(self.dim, kernel_size=3, reduction=16),
                                       nn.ReLU(inplace=True),
                                       Block(self.dim, drop_path=0),
                                       )
    def forward(self, x):
        x = self.attention(x) + x
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        embed_dim=32,       
    ):
        super(ResBlock, self).__init__()
        self.dim = embed_dim
        self.attention = nn.Sequential(Block(self.dim, drop_path=0),
                                       HERB(self.dim, kernel_size=3, reduction=16),
                                       Block(self.dim, drop_path=0),
                                       HERB(self.dim, kernel_size=3, reduction=16),
                                       Block(self.dim, drop_path=0),
                                       )
        self.con = nn.Sequential(nn.Conv2d(in_channels=self.dim, out_channels= self.dim, kernel_size=3, stride=1, padding=1, bias=False),
                                 Block(self.dim, drop_path=0),
                                 nn.Conv2d(in_channels=self.dim, out_channels= self.dim*2, kernel_size=3, stride=1, padding=1, bias=False),
                                 Block(self.dim*2, drop_path=0),
                                 Block(self.dim*2, drop_path=0),
                                 nn.Conv2d(in_channels=self.dim*2, out_channels= self.dim, kernel_size=3, stride=1, padding=1, bias=False),
                                 Block(self.dim, drop_path=0),
                                 nn.Conv2d(in_channels=self.dim, out_channels= self.dim, kernel_size=3, stride=1, padding=1, bias=False),
                                 )
        self.connect = nn.Conv2d(in_channels=self.dim * 2, out_channels= self.dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = torch.cat([self.attention(x), self.con(x)], dim=1)  
        x = self.connect(x)
        return x

class AEB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)        
        self.se = SELayer(dim)  
        self.residual = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  
        x = self.se(x)  
        x = input + x
        return x

class PCAB(nn.Module):
    def __init__(self, in_channels=1, dim=32):
        super(PCAB, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(in_channels, self.dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(self.dim, self.dim, kernel_size=5, stride=1, padding=2)
        self.aeb = AEB(dim = self.dim)
        self.fuse = nn.Conv2d(3 * self.dim, self.dim, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x1, x3, x5), dim=1)
        x = self.aeb(self.fuse(x))
        return x

class Step(nn.Module):
    def __init__(self, T, alpha_bars, rate=4, dim=32):
        super().__init__()
        self.T = T
    
        self.unet = TDB(
        in_channels=1,
        out_channels=1,
        block_out_channels=(64, 96),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        sample_size=config.para.patch_size,
    )    
        self.A = A_CDP
        self.AT = At_CDP
        self.rate = rate
        self.tau = nn.Parameter(torch.Tensor([0.5]))
        self.alpha_bars = alpha_bars
        self.dim = dim
        self.pho = nn.Parameter(torch.Tensor([0.1]))  
        self.eta = nn.Parameter(torch.Tensor([0.1]))  
        self.head = Head(in_channels=1, dim=self.dim)
        self.tail = Tail(dim=self.dim, out_channels=1)   
        self.rfmb = RFMB(embed_dim=self.dim)
        self.pcab = PCAB(in_channels=self.dim, dim=self.dim)
        self.resblock = Res(self.dim,self.dim)
        
    def forward(self, x, v, u, mask, y, t):
        t = self.T - t
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_1 = self.alpha_bars[t - 1]
        e = self.unet(x, t)
        x_0_hat = (x - e * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
        x_0_hat = self.head(x_0_hat)
        x_0_hat = self.rfmb(x_0_hat) + self.resblock(x_0_hat)
        x_0_hat = self.pcab(x_0_hat)
        x_0_hat = self.tail(x_0_hat)
        x_t_1 = torch.sqrt(alpha_bar_t_1) * x + torch.sqrt(1 - alpha_bar_t_1) * e  
        return x_t_1, v, u, x


class TPAM(nn.Module):
    def __init__(self, input_dim, dim):
        super(TPAM, self).__init__()
        self.dim = dim
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, groups=self.dim),
            nn.GELU(),
            Block(self.dim, drop_path=0),
            SELayer(self.dim, reduction=16),
            DCFB(channels=self.dim),
            nn.GELU(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, groups=self.dim),
            nn.Conv2d(in_channels=self.dim, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.fusion(x)
        return x

class Res(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1)
        self.act = nn.SELU()

    def forward(self, x):
        x = self.conv(self.act(self.conv(x)))
        return x


class DiffusionModel(nn.Module):
    def __init__(self, T, betas, layer_num=3, rate=4, dim=32):
        super().__init__()
        self.T = T
        self.dim = dim

        self.A = A_CDP
        self.AT = At_CDP
        self.layer_num = layer_num
        self.rate = rate
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars = torch.cat([torch.ones(1, device=config.para.device), self.alpha_bars])
        self.step = Step(T=self.T, alpha_bars=self.alpha_bars, rate=rate)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
        )
        self.tpam = TPAM(input_dim=self.T, dim=self.dim)


    def forward(self, x, mask, y):
        v = x.clone()  
        u = torch.zeros_like(x)
        x_out = torch.tensor([]).to(device=config.para.device)
        for t in range(self.T):
            x, v, u, x_inter = self.step(x, v, u, mask, y, t)
            x_out = torch.cat((x_out, self.conv(x_inter)), dim=1)
        x = x + self.tpam(x_out) 
        return x