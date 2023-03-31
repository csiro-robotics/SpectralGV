import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

_backbone_model_dir = os.path.join(os.path.dirname(__file__) , '../backbones/spvnas')
sys.path.append(_backbone_model_dir)

from models.backbones.spvnas.model_zoo import spvcnn
from models.aggregators.SOP import *

__all__ = ['LOGG3D']

class LOGG3D(nn.Module):
    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()

        self.spvcnn = spvcnn(output_dim=feature_dim)
        self.sop = SOP(signed_sqrt = False, do_fc=False, input_dim=feature_dim, is_tuple=False) 
   
    def forward(self, x_in):
        _, counts = torch.unique(x_in.C[:, -1], return_counts=True)
        xf, xc = self.spvcnn(x_in)
        y = torch.split(xf, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1,0,2)
        x = self.sop(x)
        return x, y[0], xc
