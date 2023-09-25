import gol
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.nn.utils.rnn import pad_sequence

from layers import GeoConv, SeqConv, VP_SDE, PointWiseFeedForward

class DiffPOI(nn.Module):
    def __init__(self, n_user, n_poi, geo_graph: Data):
        super(DiffPOI, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.local_pois = 20

        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.distance_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.temporal_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.distance_emb)
        nn.init.xavier_normal_(self.temporal_emb)

        self.geo_encoder = GeoEncoder(n_poi, self.hid_dim, geo_graph)
        self.seq_encoder = SeqEncoder(self.hid_dim)
        self.sde = VP_SDE(self.hid_dim, dt=gol.conf['dt'])
        self.ce_criteria = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1-gol.conf['keepprob'])

        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.seq_forward = PointWiseFeedForward(self.hid_dim, 0.2)

        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.geo_forward = PointWiseFeedForward(self.hid_dim, 0.2)

    def geoProp(self, poi_embs, seqs, seq_encs):
        geo_embs = self.geo_encoder.encode(poi_embs)
        if gol.conf['dropout']:
            geo_embs = self.dropout(geo_embs)

        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        geo_seq_embs = [geo_embs[seq] for seq in seqs]

        # Target-attention
        geo_embs_pad = pad_sequence(geo_seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.geo_layernorm(seq_encs.detach().unsqueeze(1))
        pad_mask = sequence_mask(seq_lengths)

        geo_embs_pad, _ = self.geo_attn(qry_embs, geo_embs_pad, geo_embs_pad, key_padding_mask=~pad_mask)
        # geo_embs_pad = geo_embs_pad + qry_embs
        geo_embs_pad = geo_embs_pad.squeeze(1)
        geo_embs_pad = self.geo_attn_layernorm(geo_embs_pad)

        geo_encs = self.geo_forward(geo_embs_pad)
        return geo_encs, geo_embs

    def seqProp(self, poi_embs, seq_graph):
        seq_embs = self.seq_encoder.encode((poi_embs, self.distance_emb, self.temporal_emb), seq_graph)
        if gol.conf['dropout']:
            seq_embs = self.dropout(seq_embs)
        seq_lengths = torch.bincount(seq_graph.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        # Self-attention
        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.seq_layernorm(seq_embs_pad)
        pad_mask = sequence_mask(seq_lengths)

        seq_embs_pad, _ = self.seq_attn(qry_embs, seq_embs_pad, seq_embs_pad, key_padding_mask=~pad_mask)
        seq_embs_pad = seq_embs_pad + qry_embs
        seq_embs_pad = self.seq_attn_layernorm(seq_embs_pad)

        seq_embs_pad = self.seq_forward(seq_embs_pad)
        seq_embs_pad = [seq[:seq_len] for seq, seq_len in zip(seq_embs_pad, seq_lengths)]

        seq_encs = torch.stack([seq.mean(dim=0) for seq in seq_embs_pad], dim=0)
        return seq_encs, seq_embs

    def sdeProp(self, geo_encs, seq_encs, target=None):
        local_embs = geo_encs
        condition_embs = seq_encs.detach()
        sde_encs = self.sde.reverse_sde(local_embs, condition_embs, gol.conf['T'])

        fisher_loss = None
        if target is not None: # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num
            mean, std = self.sde.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            perturbed_data = mean + std.unsqueeze(-1) * z
            score = - self.sde.calc_score(perturbed_data, condition_embs)
            fisher_loss = torch.square(score + z).mean()

        return sde_encs, fisher_loss

    def getTrainLoss(self, batch):
        usr, pos_lbl, _, seqs, seq_graph, cur_time = batch
        poi_embs = self.poi_emb
        if gol.conf['dropout']:
            poi_embs = self.dropout(poi_embs)

        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        sde_encs, fisher_loss = self.sdeProp(geo_encs, seq_encs, target=geo_embs[pos_lbl])

        pred_logits = seq_encs @ self.poi_emb.T + sde_encs @ geo_embs.T
        return self.ce_criteria(pred_logits, pos_lbl), fisher_loss

    def forward(self, seqs, seq_graph):
        poi_embs = self.poi_emb
        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        sde_encs, _ = self.sdeProp(geo_encs, seq_encs)

        pred_logits = seq_encs @ self.poi_emb.T + sde_encs @ geo_embs.T
        return pred_logits

class SeqEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = SeqConv(hid_dim)

    def encode(self, embs, seq_graph):
        return self.encoder(embs, seq_graph)

class GeoEncoder(nn.Module):
    def __init__(self, n_poi, hid_dim, geo_graph: Data):
        super(GeoEncoder, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.gcn_num = gol.conf['num_layer']

        edge_index, _ = add_self_loops(geo_graph.edge_index)
        dist_vec = torch.cat([geo_graph.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        dist_vec = torch.exp(-(dist_vec ** 2))
        self.geo_graph = Data(edge_index=edge_index, edge_attr=dist_vec)

        self.act = nn.LeakyReLU()
        self.geo_convs = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.geo_convs.append(GeoConv(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        geo_embs = [layer_embs]
        for conv in self.geo_convs:
            layer_embs = conv(layer_embs, self.geo_graph)
            layer_embs = self.act(layer_embs)
            geo_embs.append(layer_embs)
        geo_embs = torch.stack(geo_embs, dim=1).mean(1)
        return geo_embs


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
