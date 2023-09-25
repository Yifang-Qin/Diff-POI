import gol
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax
from torchsde import sdeint

class VP_SDE(nn.Module):
    def __init__(self, hid_dim, beta_min=0.1, beta_max=20, dt=1e-2):
        super(VP_SDE, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt

        self.score_fn = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim)
        )
        for w in self.score_fn:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def calc_score(self, x, condition):
        return self.score_fn(torch.cat((x, condition), dim=-1))
        # return self.score_fn(x)

    def forward_sde(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(gol.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def reverse_sde(self, x, condition, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            score = self.score_fn(torch.cat((x, condition), dim=-1))
            # score = self.score_fn(y)
            drift = -0.5 * beta_t * y
            drift = drift - beta_t * score
            return drift

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(gol.device)
        # ts = torch.Tensor(np.linspace(0, t, 100)).to(gol.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = torch.Tensor([log_mean_coeff]).to(x.device)
        mean = torch.exp(log_mean_coeff.unsqueeze(-1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class SDEWrapper(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'scalar'
    def __init__(self, f, g):
        super(SDEWrapper).__init__()
        self.f, self.g = f, g

    def f(self, t, y):
        return self.f(t, y)

    def g(self, t, y):
        return self.g(t, y)


class GeoConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GeoConv, self).__init__(aggr='add')
        self._cached_edge = None
        self.lin = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, geo_graph: Data):
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(geo_graph.edge_index, add_self_loops=False)
        edge_index, norm_weight = self._cached_edge
        # return x
        x = self.lin(x)

        return self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=geo_graph.edge_attr)

    def message(self, x_j, norm, dist_vec):
        return norm.unsqueeze(-1) * x_j * dist_vec.unsqueeze(-1)


class SeqConv(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(SeqConv, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        # self.alpha_slf = nn.Linear(hid_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        # nn.init.xavier_uniform_(self.alpha_slf.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, seq_graph):
        node_embs, distance_embs, temporal_embs = embs
        sess_idx, edge_index, batch_idx = seq_graph.x.squeeze(), seq_graph.edge_index, seq_graph.batch
        edge_time, edge_dist = seq_graph.edge_time, seq_graph.edge_dist

        x = node_embs[sess_idx]
        # return x
        edge_l = distance_embs[edge_dist]
        edge_t = temporal_embs[edge_time]

        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        seq_embs = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_size=edge_index.size(1))
        # seq_embs = seq_embs + self.alpha_slf(seq_embs)
        return seq_embs

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_size):
        element_sim = x_j * x_i
        src_logits = self.alpha_src(element_sim[: edge_size] + edge_l + edge_t).squeeze(-1)
        dst_logits = self.alpha_dst(element_sim[edge_size: ] + edge_l + edge_t).squeeze(-1)

        tot_logits = torch.cat((src_logits, dst_logits))
        attn_weight = softmax(tot_logits, edge_index_i)
        aggr_embs = x_j * attn_weight.unsqueeze(-1)
        return aggr_embs


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


def sequence_mask(lengths, max_len=None) -> torch.Tensor:
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
