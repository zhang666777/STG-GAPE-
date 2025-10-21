from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from typing import Dict, Any

from ...module.basic_transform import Chomp2d
from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ..base import NETWORKS
from typing import Optional
from pathlib import Path

from spikingjelly.activation_based import surrogate as sj_surrogate
from snntorch import utils
import snntorch as snn
from snntorch import surrogate
import torch
from torch import nn

from ..base import NETWORKS




# def print_tensor_stats(tensor, name="Tensor"):
#     """打印张量的形状、最小值、最大值、均值、标准差及唯一值。"""
#     print(f"{name}:")
#     print(f"  Shape: {tensor.shape}")
#     print(f"  Min: {tensor.min().item()}")
#     print(f"  Max: {tensor.max().item()}")
#     print(f"  Mean: {tensor.mean().item()}")
#     print(f"  Std: {tensor.std().item()}")
#
#     # 打印唯一值（适合稀疏张量或二值张量）
#     unique_values = tensor.unique()
#     if len(unique_values) <= 10:  # 只打印前10个唯一值，避免过多输出
#         print(f"  Unique values: {unique_values.detach().cpu().numpy()}")
#     else:
#         print(f"  Unique values: too many to display ({len(unique_values)})")


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_steps=4, grad_slope=25.0, beta=0.99, output_mems=False):
        super().__init__()
        self.spike_grad = sj_surrogate.Erf(alpha=2.0)
        self.input_size = input_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.beta = beta
        self.full_rec = output_mems
        self.lif = snn.Leaky(
            beta=self.beta,
            spike_grad=self.spike_grad,
            init_hidden=True,
            output=output_mems,
        )
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.surrogate_function1 = sj_surrogate.Erf()

    def forward(self, inputs):
        if inputs.size(-1) == self.input_size:
            # assume static spikes:
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(self.linear_ih(inputs), self.hidden_size, dim=1)
            y_hh = torch.split(self.linear_hh(h), self.hidden_size, dim=1)
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = True
        elif inputs.size(-1) == self.num_steps and inputs.size(-2) == self.input_size:
            inputs = inputs.transpose(-1, -2)  # BC, T, H
            h = torch.zeros(
                size=[inputs.shape[0], self.hidden_size, self.num_steps],
                dtype=torch.float,
                device=inputs.device,
            )
            y_ih = torch.split(
                self.linear_ih(inputs).transpose(-1, -2), self.hidden_size, dim=1
            )
            y_hh = torch.split(
                self.linear_hh(h.transpose(-1, -2)).transpose(-1, -2),
                self.hidden_size,
                dim=1,
            )
            r = self.surrogate_function1(y_ih[0] + y_hh[0])
            z = self.surrogate_function1(y_ih[1] + y_hh[1])
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
            h = (1.0 - z) * n + z * h
            cur = h
            static = False
        else:
            raise ValueError(
                f"Input size mismatch!"
                f"Got {inputs.size()} but expected (..., {self.input_size}, {self.num_steps}) or (..., {self.input_size})"
            )

        spk_rec = []
        mem_rec = []
        if self.full_rec:
            for i_step in range(self.num_steps):
                if static:
                    spk, mem = self.lif(cur)
                else:
                    spk, mem = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
                mem_rec.append(mem)
            spks = torch.stack(spk_rec, dim=-1)
            mems = torch.stack(mem_rec, dim=-1)
            return spks, mems
        else:
            for i_step in range(self.num_steps):
                if static:
                    spk = self.lif(cur)
                else:
                    spk = self.lif(cur[:, :, i_step])
                spk_rec.append(spk)
            spks = torch.stack(spk_rec, dim=-1)
            return spks

    class DeltaEncoder(nn.Module):
        def __init__(self, output_size: int):
            super().__init__()
            self.norm = nn.BatchNorm2d(1)
            self.enc = nn.Linear(1, output_size)
            self.lif = snn.Leaky(
                beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
            )

        def forward(self, inputs: torch.Tensor):
            # inputs: batch, L, C
            delta = torch.zeros_like(inputs)
            delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
            delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
            delta = self.norm(delta)
            delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
            enc = self.enc(delta)  # batch, C, L, output_size
            enc = enc.permute(0, 3, 1, 2)  # batch, output_size, C, L
            spks = self.lif(enc)
            return spks


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # batch, 1, C, L
        enc = self.encoder(inputs)  # batch, output_size, C, L
        spks = self.lif(enc)
        return spks


import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# 假设你有一个形状为 [batch_size, channels, time_steps, features] 的时序数据
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler










import torch.nn.functional as F




# 你的 Chomp2d 实现
class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size] if self.chomp_size > 0 else x


# ========= Hook 函数 =========



# ========= 网络模块 =========
class SpikeTemporalBlock2D(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        num_steps=4,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.chomp1 = Chomp2d(padding)
        self.lif1 = snn.Leaky(
            beta=0.99,
            spike_grad=sj_surrogate.Erf(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=(1, dilation),
            )
        )

        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (2, kernel_size),
                stride=stride,
                padding=(1, padding),
                dilation=(2, dilation),
            )
        )

        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.chomp2 = Chomp2d(padding)
        self.lif2 = snn.Leaky(
            beta=0.99,
            spike_grad=sj_surrogate.Erf(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=sj_surrogate.Erf(alpha=2.0),
            init_hidden=True,
            threshold=1.0,
        )

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 第一层
        out1 = self.chomp1(self.bn1(self.conv1(x)))
        spk_rec1 = [self.lif1(out1) for _ in range(self.num_steps)]
        spks1 = torch.stack(spk_rec1, dim=-1).mean(-1)

        # 第二层
        out2 = self.chomp2(self.bn2(self.conv2(spks1)))
        spk_rec2 = [self.lif2(out2) for _ in range(self.num_steps)]
        spks2 = torch.stack(spk_rec2, dim=-1).mean(-1)

        if torch.isnan(spks2).any() or torch.isinf(spks2).any():
            print("illegal value in TemporalBlock2D")

        # 残差连接
        if self.downsample is None:
            res = x
        else:
            res = self.downsample(x)
        spk_rec3 = []
        for _ in range(self.num_steps):
            res_upsampled = F.interpolate(res, size=spks2.shape[2:], mode='nearest')
            spk = self.lif(spks2 + res_upsampled)
            spk_rec3.append(spk)

        res = torch.stack(spk_rec3, dim=-1).mean(-1)

        return res






class TSSNNGRU2D(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1,
        layers: int = 1,
        num_steps: int = 4,
        grad_slope: float = 25.0,
        input_size: Optional[int] = None,
        max_length: Optional[int] = None,
        weight_file: Optional[Path] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()

        # print(f"hidden_size: {hidden_size}")

        if input_size is None:
            raise ValueError("input_size 不能为 None，请提供一个有效的整数值。")

        if encoder_type == "conv":
            self.encoder = ConvEncoder(hidden_size)
        elif encoder_type == "delta":
            self.encoder = DeltaEncoder(hidden_size)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.net = nn.Sequential(
            *[
                GRUCell(
                    hidden_size,
                    hidden_size,
                    num_steps=num_steps,
                    grad_slope=grad_slope,
                    output_mems=(i == layers - 1),
                )
                for i in range(layers)
            ]
        )
        # 新增卷积层，用于改变第二维度的大小
        self.conv = nn.Conv1d(in_channels=4, out_channels=168,
                              kernel_size=1)  # in_channels 是 spks 的通道数，out_channels 是目标大小

        self.__output_size = hidden_size * input_size

    def forward(
            self,
            inputs: torch.Tensor
    ):
        # print(f"[TSSNNGRU2D] Input shape: {inputs.shape}")

        utils.reset(self.encoder)
        for layer in self.net:
            utils.reset(layer)

        bs, length, c_num = inputs.size()
        h = self.encoder(inputs)  # B, H, C, L
        # print(f"[TSSNNGRU2D] Batch size: {bs}, Length: {length}, Channels: {c_num}")  # 打印批量大小、序列长度和通道数

        hidden_size = h.size(1)
        h = h.permute(0, 2, 3, 1).reshape(bs * c_num, length, hidden_size)  # BC, L, H
        # print(f"[TSSNNGRU2D] Reshaped output shape (h): {h.shape}")  # 打印重塑后的输出张量的形状

        for i in range(length):
            spks, mems = self.net(h[:, i, :])
            # print(f"[TSSNNGRU2D] GRU output shape at time step {i} (spks): {spks.shape}")  # 打印每个时间步的 GRU 输出形状

        spks = spks.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        # print(f"[TSSNNGRU2D] Final spks shape before transpose: {spks.shape}")  # 打印最终


        mems = mems.reshape(bs, c_num * hidden_size, -1)  # B, CH, Time Step
        # print(f"[TSSNNGRU2D] Final mems shape: {mems.shape}")  # 打印 mems 张量的形状

        output_1 = spks.transpose(1, 2)
        # 应用卷积层调整维度
        output_1 = self.conv(output_1)  # 调整通道数：4 -> 168
        # print(f"[TSSNNGRU2D] Final spks shape after convolution: {spks.shape}")  # 打印卷积后的形状

        output_2 = spks[:, :, -1]
        # print(f"[TSSNNGRU2D] Output shape 1 (spks.transpose(1, 2)): {output_1.shape}")  # 打印最终输出1的形状
        # print(f"[TSSNNGRU2D] Output shape 2 (spks[:, :, -1]): {output_2.shape}")  # 打印最终输出2的形状

        return output_1, output_2  # B * Time Step * CH, B * CH

    @property
    def output_size(self):
        return self.__output_size

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from datetime import datetime

@NETWORKS.register_module("SNN_TCN2D")

class SpikeTemporalConvNet2D(nn.Module):
    _snn_backend = "snntorch"

    def __init__(
        self,
        num_levels: int,
        channel: int,
        dilation: int,
        stride: int = 1,
        num_steps: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.2,
        max_length: int = 100,
        input_size: Optional[int] = None,
        hidden_size: int = 128,
        encoder_type: Optional[str] = "conv",
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "add",
        neuron_pe_scale: float = 1000.0,
            gru2d_params: Optional[Dict[str, Any]] = None  # 修改为可选类型
    ):
        super().__init__()

        if input_size is None:
            raise ValueError("input_size 不能为 None，请提供一个有效的整数值。")
            # 确保 gru2d_params 包含 input_size
        if gru2d_params is None:
            gru2d_params = {}
        gru2d_params['input_size'] = input_size  # 添加 input_size 到 gru2d_params

        # 检查 input_size 是否成功添加
        if 'input_size' in gru2d_params:
            print(f"input_size 成功添加到 gru2d_params: {gru2d_params['input_size']}")
        else:
            print("Warning: input_size 未成功添加到 gru2d_params！")

        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.encoder = SpikeEncoder[self._snn_backend][encoder_type](hidden_size)

        self.gru2d = TSSNNGRU2D(**gru2d_params)

        self.num_steps = num_steps
        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=num_steps,
        )
        layers = []
        num_channels = [channel] * num_levels
        num_channels.append(1)
        for i in range(num_levels + 1):
            dilation_size = dilation**i
            in_channels = hidden_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                SpikeTemporalBlock2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    num_steps=num_steps,
                )
            ]

        self.network = nn.Sequential(*layers)
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.__output_size = input_size + num_pe_neuron
        else:
            self.__output_size = input_size

    def forward(self, inputs: torch.Tensor):
        # print(f"[SNN_TCN2D] Input size: {inputs.size()}")

        utils.reset(self.encoder)
        for layer in self.network:
            utils.reset(layer)

        gru2d_output_1, gru2d_output_2 = self.gru2d(inputs)
        tcn_output = self.encoder(inputs)

        # print(f"[SNN_TCN2D] GRU2D output shape 1: {gru2d_output_1.size()}")
        # print(f"[SNN_TCN2D] GRU2D output shape 2: {gru2d_output_2.size()}")
        # print(f"[SNN_TCN2D] Encoded input size (TCN): {tcn_output.size()}")

        if self.pe_type != "none":
            tcn_output = self.pe(tcn_output.permute(1, 0, 3, 2)).permute(1, 0, 3, 2)
            # print(f"[SNN_TCN2D] Position encoded input size: {tcn_output.size()}")

        tcn_output = self.network(tcn_output)
        # print(f"[SNN_TCN2D] Network output shape: {tcn_output.size()}")

        tcn_output_squeezed = tcn_output.squeeze(1)
        # print(f"[SNN_TCN2D] Squeezed TCN output shape: {tcn_output_squeezed.size()}")

        tcn_last = tcn_output_squeezed[:, :, -1]
        # print(f"[SNN_TCN2D] TCN last time step shape: {tcn_last.shape}")

        combined_output_2d = torch.cat([gru2d_output_2, tcn_last], dim=-1)
        # print(f"[SNN_TCN2D] Combined 2D output shape: {combined_output_2d.shape}")

        tcn_output_squeezed_transposed = tcn_output_squeezed.permute(0, 2, 1)
        # #
        # print(gru2d_output_1.shape)
        # print(tcn_output_squeezed_transposed.shape)

        combined_output_3d = torch.cat([gru2d_output_1, tcn_output_squeezed_transposed], dim=-1)
        # print(f"[SNN_TCN2D] Combined 3D output shape: {combined_output_3d.shape}")

        return combined_output_2d, combined_output_3d
