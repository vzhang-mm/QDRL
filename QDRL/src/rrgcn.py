import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from src.layers import UnionRGCNLayer, UnionRGCNLayer2 #, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransE_static
from collections import defaultdict
from opt import parse_opts
from src.encoder import Transformer_encoder


args = parse_opts()

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False

        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        for i, layer in enumerate(self.layers):
            layer(g, [], r[i])
        return g.ndata.pop('h')



class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False

        return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)


    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        for i, layer in enumerate(self.layers):
            layer(g, [], r[i])
        return g.ndata.pop('h')

class Qencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes=[2,3], out_dim=256):
        super(Qencoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), out_dim)

        self.w1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=int(math.floor(3 / 2))),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, kernel_size=3, stride=1,padding=int(math.floor(3 / 2)))
        )

    def forward(self, x, e1_emb):
        # x: [B, N, D] → permute to [B, D, N] for Conv1d
        x = x.permute(0, 2, 1)

        # apply multiple convolution filters and pool
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))                 # [B, hidden_dim, N]
            p = F.adaptive_max_pool1d(c, 1)     # [B, hidden_dim, 1]
            conv_outputs.append(p.squeeze(-1))  # [B, hidden_dim]

        # concat all filters' outputs
        out = torch.cat(conv_outputs, dim=1)    # [B, hidden_dim * len(kernel_sizes)]
        out = self.fc(out)                      # [B, out_dim]
        query_emb = self.w1(torch.stack([e1_emb, out], dim=1)).squeeze(1)
        return query_emb

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels,num_words,
                 h_dim, opn, num_bases=-1,num_basis=-1,num_hidden_layers=1, dropout=0,
                 self_loop=False, skip_connect=False, layer_norm=False,
                 pre_weight=0.7, use_static=False, pre_type='TF',
                 use_cuda=False, gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        # self.sequence_len = sequence_len########################
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        # self.run_analysis = analysis
        # self.aggregation = aggregation
        # self.relation_evolve = False
        # self.weight = weight
        self.pre_weight = pre_weight
        # self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type
        # self.use_cl = use_cl
        # self.temp =temperature
        # self.angle = angle
        # self.relation_prediction = relation_prediction
        # self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.his = torch.zeros(num_ents, h_dim).float().cuda()

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            # torch.nn.init.xavier_normal_(self.words_emb)
            torch.nn.init.kaiming_normal_(self.words_emb, nonlinearity='relu')

            self.static_remb = torch.nn.Parameter(torch.Tensor(num_static_rels*2, h_dim), requires_grad=True).float()
            # torch.nn.init.xavier_normal_(self.static_remb)
            torch.nn.init.kaiming_normal_(self.static_remb, nonlinearity='relu')

            self.static_rgcn_layer = RGCNCell(num_ents,
                                             h_dim,
                                             h_dim,
                                             num_rels * 2,  # 
                                             num_bases,
                                             num_basis,
                                             num_hidden_layers,
                                             dropout,
                                             self_loop,
                                             skip_connect,
                                             encoder_name,
                                             self.opn,
                                             self.emb_rel,
                                             use_cuda,
                                             analysis)

            self.static_decoder = ConvTransE_static(h_dim)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,#
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))

        # GRU cell for relation evolving
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        
        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim)
        else:
            raise NotImplementedError

        self.MA = Transformer_encoder(d_model=self.h_dim, dim_feedforward=512, nhead=4, num_encoder_layers=2, dropout=0.2, use_query=args.add_query)

        if args.relation_prediction:
            self.query_encoder = nn.Sequential(
                nn.Linear(self.num_ents, self.h_dim),
                nn.ReLU()
            )
        else:
            self.query_encoder = nn.Sequential(
                nn.Linear(self.num_rels * 2, self.h_dim),
                nn.ReLU()
            )

        self.Qencoder = Qencoder(input_dim=256, hidden_dim=256)


    def forward(self, que_pair, sub_graph, T_idx, g_list, static_graph, use_cuda): #e_idx):
        '''
      
        '''
        # his = self.his+self.dynamic_emb
        if self.use_static:
            # his = self.dynamic_emb + self.his#
            # print(his,his.shape)
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 

            # self.statci_rgcn_layer(static_graph, [], self.static_remb)#, self.static_remb
            # static_emb = static_graph.ndata.pop('h')

            static_emb = self.static_rgcn_layer.forward(static_graph, static_graph.ndata['h'], [self.static_remb, self.static_remb])
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb

            static_emb_p = static_emb[:self.num_ents, :]
            self.h = static_emb_p
            static_remb = F.normalize(self.static_remb) if self.layer_norm else self.static_remb            
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
            static_remb = None
        '''
    
        '''
        #-------
        if args.add_his_graph:
            sub_graph = sub_graph.to(self.gpu)
            self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)
            his_r_emb = F.normalize(self.emb_rel)
            his_emb = F.normalize(self.his_ent)
        else:
            # print('不使用历史子图')
            his_r_emb = []
            his_emb = []

        his_rel_embs =[]
        '''
        g_list[0]: Graph(num_nodes=7128, num_edges=226,
        ndata_schemes={'id': Scheme(shape=(1,), dtype=torch.int64), 'norm': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'norm': Scheme(shape=(1,), dtype=torch.float32), 'type': Scheme(shape=(), dtype=torch.int64)})
        
        '''

        ### --------------
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        # print(r_len, r_idx)

        '''
       

        '''
        if args.use_onehot or not args.add_query:
            if args.relation_prediction:
                e_input = torch.zeros(self.num_ents, self.num_ents).float().cuda() if use_cuda else torch.zeros(self.num_ents,
                                                                                                                self.num_ents).float()
            else:
                e_input = torch.zeros(self.num_ents, self.num_rels * 2).float().cuda() if use_cuda else torch.zeros(self.num_ents,
                                                                                                            self.num_rels * 2).float()
            # 
            for span, e_idx in zip(r_len, uniq_e):
                if args.relation_prediction:
                    one_hot = torch.zeros(self.num_ents).to(self.gpu) if use_cuda else torch.zeros(self.num_ents)
                else:
                    one_hot = torch.zeros(self.num_rels * 2).to(self.gpu) if use_cuda else torch.zeros(self.num_rels * 2)
                rel_ids = r_idx[span[0]:span[1]]  # 
                # one_hot[rel_ids] = 1.0  # 
                #
                one_hot.index_add_(0, rel_ids, torch.ones_like(rel_ids, dtype=one_hot.dtype))
                # 
                num_rels_used = rel_ids.numel()
                if num_rels_used > 0:
                    one_hot /= num_rels_used
                e_input[e_idx] = one_hot
            query_mask = self.query_encoder(e_input)
            # print(query_mask.shape)#torch.Size([7128, 256])

        if self.pre_type=="TF":
            '''
         
            '''
            graph_embeddings = []
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)
                '''
                ##########
                '''
                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                '''
              
                '''
                graph_embeddings.append(current_h)  # current_h
            ##
            self.hr = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel

            if len(graph_embeddings) < args.train_history_len:
                embedding_dim = graph_embeddings[0].shape[-1] if graph_embeddings else self.h_dim
                num_to_pad = args.train_history_len - len(graph_embeddings)

                src_key_padding_mask = torch.cat([
                    torch.ones(graph_embeddings[0].size(0), num_to_pad, dtype=torch.bool, device=self.gpu),  # padding
                    torch.zeros(graph_embeddings[0].size(0),len(graph_embeddings), dtype=torch.bool, device=self.gpu)  # valid
                ], dim=1)

                zero_padding = torch.zeros((num_to_pad, self.num_ents, embedding_dim), device=self.gpu)
                graph_embeddings = torch.stack(graph_embeddings, dim=0) if graph_embeddings else zero_padding
                graph_embeddings = torch.cat([zero_padding, graph_embeddings], dim=0)
            else:
                # 
                graph_embeddings = torch.stack(graph_embeddings, dim=0)
                src_key_padding_mask = torch.zeros(graph_embeddings[0].size(0),len(graph_embeddings), dtype=torch.bool, device=self.gpu)   # 全部 valid

            graph_embeddings = graph_embeddings.permute(1, 0, 2) # (7128, 7, 200)


            ## ------------------
            if not args.use_onehot and args.add_query:
                # print("使用Q-encoder模块")
                if args.relation_prediction:
                    temp_r = self.h  #  
                else:
                    temp_r = self.emb_rel #

                max_rel_per_ent = max([end - start for (start, end) in r_len])

                # e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents,self.h_dim).float()
                e_input = torch.zeros(self.num_ents, max_rel_per_ent,self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents,
                                                                                              max_rel_per_ent,self.h_dim).float()

                # print(r_len, r_idx)# [113, 114]],tensor([230, 230, 230, 230, 233, 234, 23
                for span, e_idx in zip(r_len, uniq_e):
                    rel_ids = r_idx[span[0]:span[1]]  # 
                    # print(rel_ids)
                    masked_rel = temp_r[rel_ids,:]
                    num_rel = masked_rel.size(0)
                    e_input[e_idx, :num_rel, :] = masked_rel  # 
                    # x = masked_rel.sum(dim=0)  # 
                    # x = self.Agg(masked_rel)

                query_mask = torch.zeros((self.num_ents, self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
                e1_emb = self.dynamic_emb[uniq_e]
                rel_emb = e_input[uniq_e]

                query_emb = self.Qencoder(rel_emb, e1_emb)  # x: [h_dim]
                query_mask[uniq_e] = query_emb

            query = query_mask.unsqueeze(1)  # (7128, 1, 200)

            h_ = self.MA(graph_embeddings, query, src_key_padding_mask, mask=None)

            h_0 = h_[:, -1, :]
            h_0 = F.normalize(h_0) if self.layer_norm else h_0
            history_emb = h_0

            return history_emb, static_emb, static_remb, self.hr, his_emb, his_r_emb, his_rel_embs

        elif self.pre_type=="GRU":
            pass


    def predict(self,que_pair, sub_graph, T_id, test_graph, num_rels, static_graph, test_triples, use_cuda):
        
        with torch.no_grad():
            all_triples = test_triples
            # all_triples = test_triples[test_triplets[:, 0].argsort()]
            embedding, _, _, r_emb, his_emb, his_r_emb, _ = self.forward(que_pair, sub_graph, T_id, test_graph, static_graph, use_cuda)
 
            scores_ob, _ = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb, self.pre_weight, self.pre_type)
            score_seq = F.softmax(scores_ob, dim=1)
            score_en = score_seq
            scores_en = torch.log(score_en)

            return all_triples, scores_en


    def get_loss(self, que_pair, sub_graph, T_idx, glist, triples, static_triples, static_graph, use_cuda):
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)#也可作为rel loss
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        all_triples = triples

        embedding, static_emb, static_remb, r_emb, his_emb, his_r_emb, his_rel_embs = self.forward(que_pair, sub_graph, T_idx, glist, static_graph, use_cuda)
        # static_emb   torch.Size([7328,200])
        if self.use_static:
            if static_triples ==[]:
                pass#
            else:
                scores_static = self.static_decoder.forward(static_emb, static_remb, static_triples)
                # scores_static = self.static_decoder.forward(static_emb, static_triples)
                score_sti_seq = F.softmax(scores_static, dim=1)
                scores_sti_en = torch.log(score_sti_seq)
                loss_static += F.nll_loss( scores_sti_en, static_triples[:, 2])
            # print(f'static:',loss_static)

        scores_ob, _ = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb, self.pre_weight, self.pre_type)
        score_seq = F.softmax(scores_ob, dim=1)
        scores_en = torch.log(score_seq)
        '''
        '''
        if args.relation_prediction:
            loss_ent += F.nll_loss(scores_en, all_triples[:, 1])  # 
        else:
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])#
        # print(f'ent:',loss_ent)
        return loss_ent, loss_static


    def all_GCN(self, ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb),subg_index

            

    
