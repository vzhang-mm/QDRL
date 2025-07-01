import random
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
path_dir = os.getcwd()
from opt import parse_opts

args = parse_opts()


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, channels=32, kernel_size=3):

        super(ConvTransE, self).__init__()

        if args.q_type=="conv":
            self.fc = nn.Sequential(
                nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2))),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 1, kernel_size=1, stride=1)
            )
            # self.fc = UNet1D(in_channels=2, out_channels=1, features=[8, 32], d_model=embedding_dim)
        else:    
            self.w1 = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
            )

        self.pre_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, embedding, emb_rel, triplets, his_emb, pre_weight, pre_type, partial_embeding=None):
        # print(f'pre_weight,{pre_weight}')
        '''
        
        '''
        if args.relation_prediction:
            e1_embedded_all = emb_rel
        else:
            e1_embedded_all = embedding #

        embedded_his = his_emb #
        '''
           
        '''
        if args.add_his_graph:
            e1 = pre_weight * e1_embedded_all + (1 - pre_weight) * embedded_his  # 
        else:
            # print('不使用历史子图')
            e1 = e1_embedded_all

        e1_embed = embedding[triplets[:, 0]]######
        if args.relation_prediction:
            rel_embedded = embedding[triplets[:, 2]]#
        else:
            rel_embedded = emb_rel[triplets[:, 1]]

        if args.q_type=="conv":
            x = torch.stack([e1_embed, rel_embedded], dim=1)  # [B, 2, D]
            x = self.fc(x).squeeze(1)
        else:
            x = torch.cat([e1_embed, rel_embedded], 1)
            x = self.fc(x)
        
        cl_x = x
        if partial_embeding is None:
            '''
            
            '''
            # print(x.shape, e1.shape)#
            x = torch.mm(x, e1.transpose(1, 0))
            # print('x:',x.shape)##(115,7128)
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        return x, cl_x



class ConvTransE_static(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(ConvTransE_static, self).__init__()
        
        if args.q_type=="conv":
            self.fc = nn.Sequential(
            nn.Conv1d(2, 32, 3, stride=1, padding=int(math.floor(3 / 2))),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, kernel_size=3, stride=1,padding=int(math.floor(3 / 2)))
        )
        else:    
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
            )
            

    def forward(self, embedding, emb_rel, triplets):
        e1_embed = embedding[triplets[:, 0]]
        rel_embedded = emb_rel[triplets[:, 1]]

        if args.q_type=="conv":
            x = torch.stack([e1_embed, rel_embedded], dim=1)  # [B, 2, D]
            x = self.fc(x).squeeze(1)
        else:
            x = torch.cat([e1_embed, rel_embedded], 1)
            x = self.fc(x)
        x = torch.mm(x, embedding.transpose(1, 0))
        return x
        

if __name__ == '__main__':
    pass
