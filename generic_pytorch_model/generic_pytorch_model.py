from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


def from_config_to_callable(activation_name: str) -> Callable:
    if activation_name == 'LeakyReLU':
        return nn.LeakyReLU
    elif activation_name == 'Tanh':
        return nn.Tanh
    elif activation_name == 'ReLU':
        return nn.ReLU
    elif activation_name == 'Swish':
        return Swish
    elif activation_name == 'Sigmoid':
        return nn.Sigmoid
    elif activation_name == 'ELU':
        return nn.ELU
    elif activation_name == 'Hardshrink':
        return nn.Hardshrink
    elif activation_name == 'Hardsigmoid':
        return nn.Hardsigmoid
    elif activation_name == 'Hardswish':
        return nn.Hardswish
    elif activation_name == 'LogSigmoid':
        return nn.LogSigmoid
    elif activation_name == 'PReLU':
        return nn.PReLU
    elif activation_name == 'ReLU6':
        return nn.ReLU6
    elif activation_name == 'SELU':
        return nn.SELU
    elif activation_name == 'CELU':
        return nn.CELU
    elif activation_name == 'GELU':
        return nn.GELU
    elif activation_name == 'SiLU':
        return nn.SiLU
    elif activation_name == 'Mish':
        return nn.Mish
    elif activation_name == 'Softplus':
        return nn.Softplus
    elif activation_name == 'Softshrink':
        return nn.Softshrink
    elif activation_name == 'Softsign':
        return nn.Softsign
    elif activation_name == 'Tanhshrink':
        return nn.Tanhshrink
    elif activation_name == 'GLU':
        return nn.GLU
    else:
        raise ValueError(f'Activation {activation_name} not supported')

@dataclass
class NNLayerConfig:
    in_size: int
    out_size: int
    activation: Callable

    @classmethod
    def from_config(cls, config):
        return cls(
            in_size=config['in_size'],
            out_size=config['out_size'],
            activation=from_config_to_callable(config['activation'])
        )

@dataclass
class NNCOnfig:
    dropout_rate: float | None
    with_batch_norm: bool
    with_residual: bool
    layers: list[NNLayerConfig]

    @classmethod
    def from_json(cls, config):
        return cls(
            dropout_rate=config['dropout_rate'],
            with_batch_norm=config['with_batch_norm'],
            with_residual=config['with_residual'],
            layers=[NNLayerConfig.from_config(layer) for layer in config['layers']]
        )


def Swish():
    '''
    x * 1 / ( 1 + exp(-x) )
    '''
    return lambda x: x * torch.sigmoid(x)


class ConfigurableBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, activation: Callable, with_batch_norm: bool = False, with_residual: bool = False):
        super(ConfigurableBlock, self).__init__()
        self.layer = nn.Linear(in_size, out_size)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_size)
        self.with_batch_norm = with_batch_norm
        self.with_residual = with_residual

        if with_residual and in_size != out_size:
            self.projection = nn.Linear(in_size, out_size)
        else:
            self.projection = None

    def forward(self, x):
        if self.with_residual:
            residual = x
            if self.projection:
                residual = self.projection(residual)
        out = self.layer(x)
        if self.with_batch_norm:
            out_norm = self.batch_norm(out)
        out_act = self.activation(out) if not self.with_batch_norm else self.activation(out_norm)
        if self.with_residual:
            out_residual = out_act + residual
        return out_act if not self.with_residual else out_residual


class ConfigurableNN(nn.Module):
    def __init__(self, config: NNCOnfig):
        super(ConfigurableNN, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer in enumerate(config.layers):
            self.layers.append(ConfigurableBlock(layer.in_size, layer.out_size, layer.activation, config.with_batch_norm, config.with_residual))
            if i < len(config.layers) - 1 and config.dropout_rate is not None:
                self.layers.append(nn.Dropout(config.dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
