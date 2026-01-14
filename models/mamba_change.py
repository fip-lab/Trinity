import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pscan import pscan

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training


    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width



class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    

    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output








# cross attention mamba
# cross attention mamba
# cross attention mamba
# cross attention mamba
# cross attention mamba


class CrossModalMambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        
        # Audio modality MambaBlock
        self.audio_mamba = MambaBlock(config)
        
        # Visual modality MambaBlock  
        self.visual_mamba = MambaBlock(config)

        
    def forward(self, audio_x, visual_x):
        """
        Args:
            audio_x: (B, L, D) - Audio input
            visual_x: (B, L, D) - Visual input
        Returns:
            audio_output: (B, L, D) - Audio output with visual cross-attention
            visual_output: (B, L, D) - Visual output with audio cross-attention
        """
        # Get intermediate representations from both modalities
        audio_y, audio_z = self._forward_with_intermediate(self.audio_mamba, audio_x)
        visual_y, visual_z = self._forward_with_intermediate(self.visual_mamba, visual_x)
    
        # Cross-modal fusion
        # Audio output = Audio_y * Visual_z
        audio_output = audio_y * visual_z
        audio_output = self.audio_mamba.out_proj(audio_output)
        
        # Visual output = Visual_y * Audio_z  
        visual_output = visual_y * audio_z
        visual_output = self.visual_mamba.out_proj(visual_output)
        
        return audio_output, visual_output
    

    def _forward_with_intermediate(self, mamba_block, x):
        """
        Forward pass that returns intermediate y and z values
        """
        _, L, _ = x.shape

        xz = mamba_block.in_proj(x) # (B, L, 2*ED)
        x_branch, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x_branch = x_branch.transpose(1, 2) # (B, ED, L)
        x_branch = mamba_block.conv1d(x_branch)[:, :, :L] # depthwise convolution over time
        x_branch = x_branch.transpose(1, 2) # (B, L, ED)

        x_branch = F.silu(x_branch)
        y = mamba_block.ssm(x_branch)

        # z branch
        z = F.silu(z)
        
        return y, z
    



class CrossModalResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.audio_norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.visual_norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
        self.mixer = CrossModalMambaBlock(config)

    def forward(self, audio_x, visual_x):
        # Apply normalization
        audio_normed = self.audio_norm(audio_x)
        visual_normed = self.visual_norm(visual_x)
        
        # Cross-modal mixing
        audio_mixed, visual_mixed = self.mixer(audio_normed, visual_normed)
        
        # Residual connections
        audio_output = audio_mixed + audio_x
        visual_output = visual_mixed + visual_x
        
        return audio_output, visual_output


class CrossModalMamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([CrossModalResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, audio_x, visual_x):
        """
        Args:
            audio_x: (B, L, D) - Audio input
            visual_x: (B, L, D) - Visual input
        Returns:
            audio_output: (B, L, D) - Audio output
            visual_output: (B, L, D) - Visual output
        """
        for layer in self.layers:
            audio_x, visual_x = layer(audio_x, visual_x)

        return audio_x, visual_x




if __name__ == "__main__":

    # 随机种子
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def count_parameters(model):
        res = 0
        for p in model.parameters():
            if p.requires_grad:
                res += p.numel()
                # print(p)
        return res
    

    config = MambaConfig(d_model=128, n_layers=1, dt_rank='auto', d_state=16, expand_factor=2, d_conv=4)

    model = MambaBlock(config)
    # print(model, "\nNumber of parameters:", count_parameters(model))

    x = torch.randn(64, 51, 128)  # (B, L, D)
    output = model(x)
    print(output.shape)  # Should be (64, 51, 128)
    print("Output shape:", output[0, 1, :8], '\n\n')




    # Test cross-modal
    config1 = MambaConfig(d_model=128, n_layers=1, dt_rank='auto', d_state=16, expand_factor=2, d_conv=4)
    cross_modal_model = CrossModalMamba(config1)
    print("\nCross-modal model:", cross_modal_model, "\nNumber of parameters:", count_parameters(cross_modal_model))
    
    audio_output, visual_output = cross_modal_model(x, x)
    print("Audio output shape:", audio_output.shape)
    print("Visual output shape:", visual_output.shape)
    print("Audio output sample:", audio_output[0, 1, :8])
    print("Visual output sample:", visual_output[0, 1, :8])
