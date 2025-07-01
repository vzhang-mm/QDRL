import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append(".")
from src import utils
from src.utils import build_sub_graph, build_graph, get_sample_from_history_graph3
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from collections import defaultdict
from src.knowledge_graph import _read_triplets_as_list
import time

import warnings
warnings.filterwarnings('ignore')


def update_dict(subg_arr, s_to_sro, sr_to_sro,sro_to_fre, num_rels):
    # 
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)
        
def e2r(triplets, num_rels):
    # 
    src, rel, dst = triplets.transpose()
    uniq_e = np.unique(src)#

    e_to_r = defaultdict(list)#
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].append(rel)#add
    # print(e_to_r)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        # print(e_to_r[e])
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])

    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()

    return [uniq_e, r_len, r_idx]



def e2t(triplets, num_rels):
    # triplets: numpy array of shape (N, 3), each row is (head, rel, tail)
    src, rel, dst = triplets.transpose()

    uniq_e = np.unique(src)  # 
    e_to_t = defaultdict(list)

    for j, (h, r, t) in enumerate(triplets):
        e_to_t[h].append(t)  #

    t_len = []
    t_idx = []
    idx = 0
    for e in uniq_e:
        ts = e_to_t[e]
        t_len.append((idx, idx + len(ts)))  # 
        t_idx.extend(ts)
        idx += len(ts)

    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()       # 
    t_len = torch.from_numpy(np.array(t_len)).long().cuda()         # 
    t_idx = torch.from_numpy(np.array(t_idx)).long().cuda()         # 

    return [uniq_e, t_len, t_idx]


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    input_list = [snap for snap in history_list[-args.train_history_len:]]

    if args.add_his_graph:
        his_list = history_list[:]
        subg_arr = np.concatenate(his_list)
        sr_to_sro = np.load('./data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()
    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples = test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        if args.relation_prediction:
            que_pair = e2t(test_snap, num_rels)
            que_pair_inv = e2t(inverse_triples, num_rels)
        else:
            que_pair = e2r(test_snap, num_rels)
            que_pair_inv = e2r(inverse_triples, num_rels)
        '''
    
        '''
        if args.add_his_graph:
            sub_snap, sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap, num_nodes, num_rels,use_cuda, args.gpu)
        else:
            sub_snap, sub_snap_inv = [],[]

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)

        test_triples, final_score = model.predict(que_pair, sub_snap, time_idx, history_glist, num_rels, static_graph, test_triples_input, use_cuda)
        inv_test_triples, inv_final_score = model.predict(que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, use_cuda)

        # print(f'test_triples, final_score:{test_triples.shape},{ final_score.shape}')#(211,3) (211,7128)
        if args.relation_prediction:
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score,all_ans_r_list[time_idx],eval_bz=1000,rel_predict=1)
            mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples,inv_final_score,all_ans_r_list[time_idx],eval_bz=1000,rel_predict=1)
        else:
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)

        input_list.pop(0)
        input_list.append(test_snap)
            
        
        idx += 1

    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")#############
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)

    print("(all_raw) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,5):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))
    
    # 
    if mode == "test": 
        filename = './result/'+ args.dataset + ".csv"
        if os.path.isfile(filename) == False:# 
            with open (filename,'w', newline='') as f:
                fieldnames=['encoder','opn','pre_type','use_static','use_attr','gpu','datetime','pre_weight',
                            'train_len','lr','n_hidden',
                            'filter_MRR','filter_H@1','filter_H@3','filter_H@10',
                            'filter_inv_MRR','filter_inv_H@1','filter_inv_H@3','filter_inv_H@10',
                            'all_MRR','all_H@1','all_H@3','all_H@10',
                            'filter_all_MRR','filter_all_H@1','filter_all_H@3','filter_all_H@10']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        with open (filename,'a', newline='') as f:
            writer = csv.writer(f)
            row={'encoder':args.encoder,'opn':args.opn,'pre_type':args.pre_type,'use_static':args.add_static_graph,'gpu':args.gpu,'datetime':datetime.now(),'pre_weight':args.pre_weight,
                'train_len':args.train_history_len,'lr':args.lr,'n_hidden':args.n_hidden,
                'filter_MRR':float(mrr_filter),'filter_H@1':hit_filter[0],'filter_H@3':hit_filter[1],'filter_H@10':hit_filter[2],
                'filter_inv_MRR':float(mrr_filter_inv),'filter_inv_H@1':hit_filter_inv[0],'filter_inv_H@3':hit_filter_inv[1],'filter_inv_H@10':hit_filter_inv[2],
                'all_MRR':all_mrr_raw.item(),'all_H@1':all_hit_raw[0],'all_H@3':all_hit_raw[1],'all_H@10':all_hit_raw[2],
                'filter_all_MRR':all_mrr_filter.item(),'filter_all_H@1':all_hit_filter[0],'filter_all_H@3':all_hit_filter[1],'filter_all_H@10':all_hit_filter[2]}
            writer.writerow(row.values())
            
    return all_mrr_raw, all_mrr_filter
    

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)############################
    valid_list = utils.split_by_time(data.valid)#
    test_list = utils.split_by_time(data.test)


    print(f'train_list:{len(train_list),train_list[0].shape,}')#train_list:(304, (113, 3))

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.pre_weight, args.pre_type,  args.n_hidden, args.encoder,str(time.time()))
    model_state_file = './models/' + model_name+ ".pt"
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("./data/" + args.dataset + "/e-w-graph-llm.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = np.max(static_triples[:, 2])+1 - data.num_nodes####
        print(np.max(static_triples[:, 2])+1)
        print(f'*************{num_static_rels},{num_words}')
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, inverse_static_triples, static_graph = 0, 0, [], [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          args.n_hidden,
                          args.opn,
                          # sequence_len=args.train_history_len,
                          num_bases=args.n_bases,#####
                          num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers,
                          dropout=args.dropout,
                          self_loop=args.self_loop,
                          skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm,

                          # input_dropout=args.input_dropout,
                          # hidden_dropout=args.hidden_dropout,
                          # feat_dropout=args.feat_dropout,
                          # aggregation=args.aggregation,
                          # weight=args.weight,

                          pre_weight = args.pre_weight,
                          # discount=args.discount,
                          # angle=args.angle,
                          use_static=args.add_static_graph,###
                          pre_type = args.pre_type,
                          # use_cl = args.use_cl,
                          # temperature = args.temperature, ######
                          # entity_prediction=args.entity_prediction,
                          # relation_prediction=args.relation_prediction,#
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          analysis=args.run_analysis
                          )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        '''

        '''
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

        # inverse_static_triples = static_triples[:, [2, 1, 0]]
        # inverse_static_triples[:, 1] = inverse_static_triples[:, 1] + num_static_rels
        # print(inverse_static_triples)

        static_triples = torch.from_numpy(static_triples).long().cuda()
        # inverse_static_triples = torch.from_numpy(inverse_static_triples).long().cuda()
        inverse_static_triples = []
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # # 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # #

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter= test(model,
                                train_list+valid_list, 
                                test_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                all_ans_list_test, 
                                all_ans_list_r_test, 
                                model_state_file, 
                                static_graph, 
                                "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        his_best = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_m = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]

            for train_sample_num in tqdm(idx):
            # for train_sample_num in idx:
                if train_sample_num == 0:
                    continue
                output = train_list[train_sample_num:train_sample_num+1]#his_emb
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]#history_glist
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]
                '''
               
                '''
                if args.add_his_graph:
                    subgraph_arr = np.load('./data/{}/his_graph_for/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                    subgraph_arr_inv = np.load('./data/{}/his_graph_inv/train_o_r_{}.npy'.format(args.dataset, train_sample_num))

                    subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu) #取出采样子图
                    subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)
                else:
                    subg_snap, subg_snap_inv = [],[]

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                '''
                '''

                if args.relation_prediction:
                    que_pair = e2t(output[0], num_rels)
                    que_pair_inv = e2t(inverse_triples, num_rels)
                else:
                    que_pair = e2r(output[0], num_rels)
                    que_pair_inv = e2r(inverse_triples, num_rels)

                # print(que_pair)
                # generate history graph->input_list
                ''' 
                
                '''
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda()
                # 
                for id in range(2): 
                    if id %2 ==0:
                        # 
                        loss_e, loss_static = model.get_loss(que_pair, subg_snap, train_sample_num,
                                                                              history_glist, triples,
                                                                              static_triples, static_graph, use_cuda)
                    else:
                        #
                        loss_e, loss_static = model.get_loss(que_pair_inv, subg_snap_inv, train_sample_num,
                                                                              history_glist, inverse_triples,
                                                                              inverse_static_triples, static_graph, use_cuda)
                    if loss_static.item() > 3:
                        loss = 0.9*loss_e + 0.1*loss_static
                    else:
                        loss = loss_e
                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_static.append(loss_static.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # 
                    optimizer.step()
                    # scheduler.step()  ##
                    optimizer.zero_grad()
                # break
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, 
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph, 
                                    mode="train")
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            break
                        if his_best>=5:
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            torch.cuda.empty_cache()

            if epoch >= 5:
                mrr_raw, mrr_filter = test(model,
                                    train_list+valid_list,
                                    test_list,
                                    num_rels,
                                    num_nodes,
                                    use_cuda,
                                    all_ans_list_test,
                                    all_ans_list_r_test,
                                    model_state_file,
                                    static_graph,
                                    mode="test")

    return mrr_raw, mrr_filter

from opt import parse_opts
if __name__ == '__main__':
    args = parse_opts()
    print(args)
    run_experiment(args)


# python main.py -d ICEWS14 --train-history-len 7  --lr 0.0005 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 256 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --pre-weight 0.7  --pre-type TF --self-loop --add-static-graph --q-type conv --add_query

# --relation-prediction 关系预测

