import torch
import numpy as np
from torch.utils.data import Dataset
import gol
from gol import pLog
from os.path import join, exists
import pickle as pkl
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, coalesce


def getSeqGraph(seq, time_list):
    i, x, senders, nodes = 0, [], [], {}
    for node in seq:
        if node not in nodes:
            nodes[node] = i
            x.append([node])
            i += 1
        senders.append(nodes[node])
    x = torch.LongTensor(x)
    edge_index = torch.LongTensor([senders[: -1], senders[1:]])


    def get_min(interv):
        interv_min = interv.clone()
        interv_min[interv_min == 0] = 2 ** 31
        return interv_min.min()

    time_interv = (time_list[1:] - time_list[:-1]).long()
    dist_interv = gol.dist_mat[seq[:-1], seq[1:]].long()
    mean_interv = dist_interv.float().mean()
    if time_interv.size(0) > 0:
        time_interv = torch.clamp((time_interv / get_min(time_interv)).long(), 0, gol.conf['interval'] - 1)
        dist_interv = torch.clamp((dist_interv / get_min(dist_interv)).long(), 0, gol.conf['interval'] - 1)
    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes), mean_interv=mean_interv, edge_time=time_interv, edge_dist=dist_interv)


class GraphData(Dataset):
    def __init__(self, n_user, n_poi, seq_data, pos_dict, is_eval=False, tr_dict=None):
        self.n_user, self.n_poi = n_user, n_poi
        self.seq_data = seq_data
        self.is_eval = is_eval

        self.tr_dict = tr_dict
        self.pos_dict = pos_dict
        self.userSet = list(self.pos_dict.keys())
        self.len = len(self.seq_data)
        self.max_len = gol.conf['max_len']

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.is_eval:
            uid, poi, seq, seq_time, cur_time, _ = self.seq_data[index]
            pos_set = set(self.pos_dict[uid])
            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]
                seq_time = seq_time[-self.max_len:]
            seq_time = torch.LongTensor(seq_time)
            seq_graph = getSeqGraph(seq, seq_time)

            seq = torch.LongTensor(seq)
            neg = np.random.randint(0, self.n_poi)
            while neg in pos_set:
                neg = np.random.randint(0, self.n_poi)
            return uid, poi, neg, seq, seq_graph, (cur_time // 60) % 168
        else:
            uid, poi, seq, seq_time, cur_time, _ = self.seq_data[index]
            labels = torch.zeros((self.n_poi, )).long()
            labels[poi] = 1

            exclude_set = torch.LongTensor(list(set(self.tr_dict[uid])))
            exclude_mask = torch.zeros((self.n_poi, )).bool()
            exclude_mask[exclude_set] = 1
            exclude_mask[poi] = 0

            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]
                seq_time = seq_time[-self.max_len:]
            seq_time = torch.LongTensor(seq_time)
            seq_graph = getSeqGraph(seq, seq_time)
            seq = torch.LongTensor(seq)

            return uid, labels.unsqueeze(0), exclude_mask.bool().unsqueeze(0), \
                seq, seq_graph, (cur_time // 60) % 168


def collate_edge(batch):
    u, p, n, s, s_graph, t = tuple(zip(*batch))
    u = torch.LongTensor(u).to(gol.device)
    p = torch.LongTensor(p).to(gol.device)
    n = torch.LongTensor(n).to(gol.device)
    s_graph = Batch.from_data_list(s_graph).to(gol.device)
    t = torch.LongTensor(t).to(gol.device)
    return u, p, n, s, s_graph, t

def collate_eval(batch):
    u, label, exclude_mask, seq, seq_graph, t = tuple(zip(*batch))
    u = torch.LongTensor(u).to(gol.device)
    s_graph = Batch.from_data_list(seq_graph).to(gol.device)
    t = torch.LongTensor(t).to(gol.device)
    return u, torch.cat(label, dim=0), torch.cat(exclude_mask, dim=0), seq, s_graph, t

def getDatasets(path='../data/processed', dataset='foursquare'):
    dist_pth = join(path, dataset)
    gol.pLog(f'Loading from {dist_pth}')
    with open(join(dist_pth, 'all_data.pkl'), 'rb') as f:
        n_user, n_poi = pkl.load(f)
        _ = pkl.load(f)
        trn_set, val_set, tst_set = pkl.load(f)
        trn_df, val_df, tst_df = pkl.load(f)

    trn_dict, val_dict, tst_dict = {}, {}, {}
    for uid, line in trn_df.groupby('uid'):
        trn_dict[uid] = line['poi'].tolist()
    for uid, line in val_df.groupby('uid'):
        val_dict[uid] = line['poi'].tolist()
    for uid, line in tst_df.groupby('uid'):
        tst_dict[uid] = line['poi'].tolist()

    trn_ds = GraphData(n_user, n_poi, trn_set, trn_dict)
    val_ds = GraphData(n_user, n_poi, val_set, val_dict, is_eval=True, tr_dict=trn_dict)
    tst_ds = GraphData(n_user, n_poi, tst_set, tst_dict, is_eval=True, tr_dict=trn_dict)

    with open(join(dist_pth, 'dist_graph.pkl'), 'rb') as f:
        geo_edges = torch.LongTensor(pkl.load(f))
    edge_weights = torch.Tensor(np.load(join(dist_pth, 'dist_on_graph.npy')))
    edge_weights /= edge_weights.max()
    geo_edges, edge_weights = to_undirected(geo_edges, edge_weights, num_nodes=n_poi)

    assert geo_edges.size(1) == edge_weights.size(0)
    assert len(trn_df) == len(trn_set) + n_user

    geo_graph = Data(edge_index=geo_edges, edge_attr=edge_weights).to(gol.device)
    pLog(f'{dataset.upper()} loaded. #User: {n_user}, #POI: {n_poi}')
    pLog(f'#Train: {len(trn_set)}, #Valid: {len(val_set)}, #Test: {len(tst_set)}')
    pLog(f'Geograph #edge: {geo_edges.size(1) // 2}, Avg. degree: {geo_edges.size(1) / n_poi / 2:.2f}')
    return n_user, n_poi, (trn_ds, val_ds, tst_ds), geo_graph
