import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
import sys
import numpy as np
from Mymodel import SinusoidalTimeEncoding
from Mymodel import SETimeFusion
from Mymodel import MambaLSTM

class MambaLSTMPredictor(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, pre_len, drop_rate, drop_path_rate):
        super().__init__()

        self.states = None
        self.dynamic_proj = nn.Sequential(
            nn.Linear(in_chans, embed_dim*2),
            nn.GELU(),
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.MambaLSTM = MambaLSTM(
            img_size, patch_size, embed_dim, embed_dim, depths,drop_rate, drop_path_rate
        )
        self.fc = nn.Linear(embed_dim, pre_len)
        self.time_embed_pro = nn.Linear(32, embed_dim)
        self.time_embed = SinusoidalTimeEncoding(embed_dim) 
        # self.se_time_fusion = SETimeFusion(time_dim=embed_dim, embed_dim=embed_dim)  
        self.se_time_fusion = SETimeFusion(embed_dim) 

    def forward(self, x, time_features, targets_len=1):
        x=x.permute(0,1,3,4,2)
        x=self.dynamic_proj(x)
        x=x.permute(0,1,4,2,3)
        if self.states != None:
            self.states = None   
        long_out=[]
        outputs = []
        inputs_len = x.shape[1]
        assert time_features.shape[1] == inputs_len + targets_len, f"{time_features.shape[1]}"
        last_input = x[:, -1] 
        time_embeding=self.time_embed(time_features)
        for i in range(6):
            x_step=x[:, i]
            t_step = time_embeding[:, i]
            x_step = self.se_time_fusion(x_step, t_step)
            output, self.states = self.MambaLSTM(x_step, self.states)
            long_out.append(output)
        for i in range(6,inputs_len - 1):
            x_step=x[:, i]
            t_step = time_embeding[:, i]  
            x_step = self.se_time_fusion(x_step, t_step)
            output, self.states = self.MambaLSTM(x_step, self.states)
            outputs.append(output)
        t_step = time_embeding[:, inputs_len-1]  
        last_input =self.se_time_fusion(last_input, t_step)
        output, self.states = self.MambaLSTM(last_input, self.states)
        dec_out=output
        out = self.fc(dec_out.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return out

class Main_MambaLSTM(nn.Module):
    def __init__(self, 
                 c_h, c_w, c_static_feat, c_semantic_mats, c_grid_node_map, c_valid_mask,
                 input_dim=4, lstm_hidden=64, pre_len=1):
        super().__init__()
        self.Main_MambaLSTM = MambaLSTMPredictor(img_size=c_h, patch_size=1,in_chans=input_dim,embed_dim=lstm_hidden, 
                                         depths=2, pre_len=pre_len, drop_rate=0.0, drop_path_rate=0.1)

    def forward(self, input,time_feature):
        out=self.Main_MambaLSTM(input,time_feature)
        return out


