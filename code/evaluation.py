import numpy as np
import torch
import metrics
from torch.utils.data import DataLoader
import gol
from model import DiffPOI
from dataset import GraphData, collate_eval, collate_edge

Ks = [1, 2, 5, 10, 20, 50]

def eval_one_user(X):
    score, label = X
    ranked_idx = np.argsort(-score)[: max(Ks)]
    rank_results = label[ranked_idx]

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'mrr': 0.}
    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(metrics.precision_at_k(rank_results, K))
        recall.append(metrics.recall_at_k(rank_results, K, label.sum()))
        ndcg.append(metrics.ndcg_at_k(rank_results, K))
        hit_ratio.append(metrics.hit_at_k(rank_results, K))
    mrr = metrics.mrr(rank_results)

    result['precision'] += precision
    result['recall'] += recall
    result['ndcg'] += ndcg
    result['hit_ratio'] += hit_ratio
    result['mrr'] += mrr
    return result


def eval_model(model: DiffPOI, eval_set: GraphData):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'mrr': 0.}
    result_user = [{'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'mrr': 0.} for _ in range(4)]
    usr_cnt = [1 for _ in range(4)]
    eval_loader = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_eval)

    with torch.no_grad():
        model.eval()
        test_outputs, tot_cnt = [], 0
        for idx, batch in enumerate(eval_loader):
            u, pos_list, exclude_mask, seqs, seq_graph, cur_time = batch
            item_score = model(seqs, seq_graph)

            usr_lbl = torch.zeros_like(u)
            seq_graph.mean_interv[seq_graph.mean_interv.isnan()] = seq_graph.mean_interv[~seq_graph.mean_interv.isnan()].mean()
            usr_lbl[seq_graph.mean_interv > 5] += 1
            usr_lbl[seq_graph.mean_interv > 10] += 1
            usr_lbl[seq_graph.mean_interv > 15] += 1

            item_score[exclude_mask] = -1e10
            item_score = item_score.cpu()

            for score, label, usr_tp in zip(item_score, pos_list, usr_lbl):
                ranked_idx = np.argsort(-score)[: max(Ks)]
                rank_results = label[ranked_idx]

                precision, recall, ndcg, hit_ratio = [], [], [], []
                for K in Ks:
                    precision.append(metrics.precision_at_k(rank_results, K))
                    recall.append(metrics.recall_at_k(rank_results, K, 1))
                    ndcg.append(metrics.dcg_at_k(rank_results, K)) # Keep aligned with GSTN
                    # ndcg.append(metrics.ndcg_at_k(rank_results, K))
                    hit_ratio.append(metrics.hit_at_k(rank_results, K))
                mrr = metrics.mrr(rank_results)

                result['precision'] += precision
                result['recall'] += recall
                result['ndcg'] += ndcg
                result['hit_ratio'] += hit_ratio
                result['mrr'] += mrr
                tot_cnt += 1

                result_user[usr_tp]['precision'] += precision
                result_user[usr_tp]['recall'] += recall
                result_user[usr_tp]['ndcg'] += ndcg
                result_user[usr_tp]['hit_ratio'] += hit_ratio
                result_user[usr_tp]['mrr'] += mrr
                usr_cnt[usr_tp] += 1

    for tp in range(4):
        result_user[tp]['precision'] /= usr_cnt[tp]
        result_user[tp]['recall'] /= usr_cnt[tp]
        result_user[tp]['ndcg'] /= usr_cnt[tp]
        result_user[tp]['hit_ratio'] /= usr_cnt[tp]
        result_user[tp]['mrr'] /= usr_cnt[tp]

    result['precision'] /= tot_cnt
    result['recall'] /= tot_cnt
    result['ndcg'] /= tot_cnt
    result['hit_ratio'] /= tot_cnt
    result['mrr'] /= tot_cnt
    return result, result_user
