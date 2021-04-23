'''
Author: Ernie Chu
Project name: TSM-Net
Last Modify date: 2021/3/26
'''

import torch
import torch.nn as nn
import torch.functional as F
import math

class ConvBlock(nn.Module):
    '''
    1D CNN Block including configurations for
    - in_channel
    - out_channel
    - kernel_size
    - padding
    - stride
    - normalization layer
    - dropout rate
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        padding: int = 0,
        stride: int = 1,
        norm_layer = nn.InstanceNorm1d,
        dropout: float = 0.
    ):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            stride=stride
        )]
        if norm_layer:
            layers.append(norm_layer(out_channels))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
            
class ContentEncoder(nn.Module):
    '''
    Encoder for music content except tempo information using ConvBlock
    '''
    def __init__(
        self,
        in_channels=2,
        out_channels=1024,
        base_channels=32,
        block = ConvBlock
    ):
        super(ContentEncoder, self).__init__()
        layers = [block(in_channels, base_channels)]
        for channel_multiplier in range(0, int(math.sqrt(out_channels/base_channels))):
            layers.append(
                block(
                    base_channels*2**channel_multiplier,
                    base_channels*2**(channel_multiplier+1)
                )
            )
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class TempoEmbedding(nn.Module):
    '''
    Neural embedding for the tempo
    '''
    def __init__(
        self,
        in_channels=1
    ):
        super(TempoEmbedding, self).__init__()

if __name__ == "__main__":
    model = ContentEncoder()
    print(model)