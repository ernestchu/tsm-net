'''
Author: Ernie Chu
Project name: TSM-Net
Last Modify date: 2021/3/26
'''

import torch
import torch.nn as nn
import torch.functional as F
import math

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=nn.InstanceNorm1d, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv1d(in_size, out_size, 4, 2, 1, bias=False)]
        if norm_layer:
            layers.append(norm_layer(out_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, norm_layer=nn.InstanceNorm1d, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose1d(in_size, out_size, 4, 2, 1, bias=False)]
        if norm_layer:
            layers.append(norm_layer(out_size))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        if x.shape[-1] != skip_input.shape[-1]:
            x = F.pad(x, (0, 1))
        x = torch.cat((x, skip_input), 1)

        return x

class ContentEncoder(nn.Module):
    '''
    Encoder for music content except tempo information using ConvBlock
    '''
    def __init__(self, in_channels=2, out_channels=2):
        super(ContentEncoder, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, dropout=0.2)
        self.down6 = UNetDown(512, 512, dropout=0.2)
        self.down7 = UNetDown(512, 512, dropout=0.2)
        self.down8 = UNetDown(512, 512, dropout=0.2)

        self.up1 = UNetUp(512, 512, dropout=0.2)
        self.up2 = UNetUp(1024, 512, dropout=0.2)
        self.up3 = UNetUp(1024, 512, dropout=0.2)
        self.up4 = UNetUp(1024, 512, dropout=0.2)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, out_channels, 4, padding=1),
            nn.ConstantPad1d((0, 1), 0),
        )
        
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
    
class TempoEmbedding(nn.Module):
    '''
    Neural embedding for the tempo
    '''
    def __init__(self, in_channels=1, out_channels=1):
        super(TempoEmbedding, self).__init__()

        self.down1 = UNetDown(in_channels, 64, norm_layer=0)
        self.down2 = UNetDown(64, 128, norm_layer=0)
        self.down3 = UNetDown(128, 256, norm_layer=0)
        self.down4 = UNetDown(256, 512, norm_layer=0)
        self.down5 = UNetDown(512, 512, dropout=0.2, norm_layer=0)
        self.down6 = UNetDown(512, 512, dropout=0.2, norm_layer=0)
        self.down7 = UNetDown(512, 512, dropout=0.2, norm_layer=0)
        self.down8 = UNetDown(512, 512, dropout=0.2, norm_layer=0)
            
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        return self.final(d8)
        
class TSM_NetTemporal(nn.Module):
    '''
    Neural embedding for the tempo
    '''
    def __init__(self):
        super(TSM_NetTemporal, self).__init__()
        self.content_encoder = ContentEncoder()
        self.tempo_embedding = TempoEmbedding()
        
        
if __name__ == "__main__":
    model = ContentEncoder()
    print(model)