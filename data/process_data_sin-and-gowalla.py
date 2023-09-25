import os.path
import random
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from math import cos, asin, sqrt, pi
from os.path import join


def distance(lat1, lon1, lat2, lon2):
    r = 6371
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))

def distance_mat_form(lat_vec: np.ndarray, lon_vec: np.ndarray):
    # Shape of lat_vec & lon_vec: [n_poi, 1]
    r = 6371
    p = pi / 180
    lat_mat = np.repeat(lat_vec, lat_vec.shape[0], axis=-1)
    lon_mat = np.repeat(lon_vec, lon_vec.shape[0], axis=-1)
    a_mat = 0.5 - np.cos((lat_mat.T - lat_mat) * p) / 2 \
            + np.matmul(np.cos(lat_vec * p), np.cos(lat_vec * p).T) * (1 - np.cos((lon_mat.T - lon_mat) * p)) / 2
    return 2 * r * np.arcsin(np.sqrt(a_mat))

def gen_nei_graph(df: pd.DataFrame, n_users, n_pois, train=False):
    nei_dict = {idx: [] for idx in range(n_users)}
    edges = [[], []]

    for _uid, _item in df.groupby('uid'):
        _val_piv, _ = np.quantile(_item['time'], [0.8, 0.9])
        poi_list = _item['poi'].tolist()
        if train:
            poi_list = poi_list[: _val_piv]

        nei_dict[_uid] = poi_list
        edges[0] += [_uid for _ in poi_list]
        edges[1] += poi_list

    return nei_dict, torch.LongTensor(edges)

def gen_loc_graph(poi_loc, n_pois, thre, _dist_mat=None):
    if _dist_mat is None:
        lat_vec = np.array([poi_loc[poi][0] for poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)
        lon_vec = np.array([poi_loc[poi][1] for poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)
        _dist_mat = distance_mat_form(lat_vec, lon_vec)

    adj_mat = np.triu(_dist_mat <= thre, k=1)
    num_edges = adj_mat.sum()
    print(f'Edges on dist_graph: {num_edges}, avg degree: {num_edges / n_pois}')

    idx_mat = np.arange(n_pois).reshape(-1, 1).repeat(n_pois, -1)
    row_idx = idx_mat[adj_mat]
    col_idx = idx_mat.T[adj_mat]
    edges = np.stack((row_idx, col_idx))

    nei_dict = {poi: [] for poi in range(n_pois)}
    for e_idx in range(edges.shape[1]):
        src, dst = edges[:, e_idx]
        nei_dict[src].append(dst)
        nei_dict[dst].append(src)
    return _dist_mat, edges, nei_dict

def remap(df: pd.DataFrame, n_users, n_pois):
    uid_dict = dict(zip(pd.unique(df['uid']), range(n_users)))
    poi_dict = dict(zip(pd.unique(df['poi']), range(n_pois)))
    df['uid'] = df['uid'].map(uid_dict)
    df['poi'] = df['poi'].map(poi_dict)
    return df, uid_dict, poi_dict

random.seed(1234)
target_dataset = 'Foursquare' # ['Foursquare', 'Gowalla'] "Foursquare" denotes the Singapore dataset
# target_dataset = 'Gowalla' # ['Foursquare', 'Gowalla']
source_pth = f'./raw/poidata/{target_dataset}'

dist_pth = f'./processed/{target_dataset.lower()}'
col_names = ['uid', 'poi', 'latlon', 'daytime', 'date']
review_pth = join(dist_pth, 'all_data.pkl')

if not os.path.exists(review_pth) or True:
    print(f'Load from {source_pth}\nData preprocessing...')
    trn_df = pd.read_csv(join(source_pth, 'train.txt'), sep='\t', header=None, names=col_names)
    val_df = pd.read_csv(join(source_pth, 'tune.txt'), sep='\t', header=None, names=col_names)
    tst_df = pd.read_csv(join(source_pth, 'test.txt'), sep='\t', header=None, names=col_names)
    review_df = pd.concat((trn_df, val_df, tst_df))
    review_df.drop(review_df[review_df['poi'] == 'LOC_null'].index, inplace=True)

    review_df['lat'] = review_df['latlon'].str.split(',', expand=True)[0].astype(np.float64)
    review_df['lon'] = review_df['latlon'].str.split(',', expand=True)[1].astype(np.float64)

    # 5-core cleaning
    user_remove, poi_remove = [], []
    for poi, line in review_df.groupby('poi'):
        if len(line) < 5:
            poi_remove.append(poi)
    for user, line in review_df.groupby('uid'):
        if len(line) < 5:
            user_remove.append(user)

    for uid in user_remove:
        review_df.drop(review_df[review_df['uid'] == uid].index, inplace=True)
    for poi in poi_remove:
        review_df.drop(review_df[review_df['poi'] == poi].index, inplace=True)

    n_user, n_poi = pd.unique(review_df['uid']).shape[0], pd.unique(review_df['poi']).shape[0]
    review_df, uid_dic, poi_dic = remap(review_df, n_user, n_poi)# .iloc[:20]
    review_df['lat'] = review_df['latlon'].str.split(',', expand=True)[0].astype(np.float64)
    review_df['lon'] = review_df['latlon'].str.split(',', expand=True)[1].astype(np.float64)

    # Get time order
    review_df['time'] = review_df['daytime'].str.split(':', expand=True)[0].astype(int) * 60
    review_df['time'] += review_df['daytime'].str.split(':', expand=True)[1].astype(int)
    review_df['time'] += review_df['date'].astype(int) * 60 * 24
    review_df['date'] = review_df['date'].astype(int)
    review_df.sort_values(by='time', inplace=True)
    review_df = review_df[['uid', 'poi', 'lat', 'lon', 'time', 'date']]

    # Generate data
    trn_df, val_df, tst_df = [], [], []
    trn_set, val_set, tst_set = [], [], []
    for uid, line in review_df.groupby('uid'):
        val_piv, test_piv = np.quantile(line['time'], [0.8, 0.9])
        trn_df.append(line[line['time'] < val_piv])
        val_df.append(line[(line['time'] >= val_piv) & (line['time'] < test_piv)])
        tst_df.append(line[line['time'] >= test_piv])

        pos_list, time_list, date_list = line['poi'].tolist(), line['time'].tolist(), line['date'].tolist()
        for i in range(1, len(pos_list)):
            location = (line['lat'].iloc[i], line['lon'].iloc[i])
            if time_list[i] < val_piv:
                trn_set.append((uid, pos_list[i], pos_list[: i], time_list[:i], time_list[i], date_list[i]))
            elif val_piv <= time_list[i] < test_piv:
                val_set.append((uid, pos_list[i], pos_list[: i], time_list[:i], time_list[i], date_list[i]))
            else:
                tst_set.append((uid, pos_list[i], pos_list[: i], time_list[:i], time_list[i], date_list[i]))

    trn_df, val_df, tst_df = pd.concat(trn_df), pd.concat(val_df), pd.concat(tst_df)

    # get loc_dict
    loc_dict = {poi: None for poi in range(n_poi)}
    for poi, item in review_df.groupby('poi'):
        lat, lon = item['lat'].iloc[0], item['lon'].iloc[0]
        loc_dict[poi] = (lat, lon)

    print('trn_set size:', len(trn_set))
    print('val_set size:', len(val_set))
    print('tst_set size:', len(tst_set))
    print('trn_df size:', len(trn_df))
    print('val_df size:', len(val_df))
    print('tst_df size:', len(tst_df))
    print('review_df size:', len(review_df))

    if not os.path.exists(dist_pth):
        os.mkdir(dist_pth)
    with open(review_pth, 'wb') as f:
        pkl.dump((n_user, n_poi), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(review_df, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump((trn_set, val_set, tst_set), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump((trn_df, val_df, tst_df), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(loc_dict, f, pkl.HIGHEST_PROTOCOL)
    print(f'Process Done\n')

with open(review_pth, 'rb') as f:
    n_user, n_poi = pkl.load(f)
    review_df = pkl.load(f)
    trn_set, val_set, tst_set = pkl.load(f)
    trn_df, val_df, tst_df = pkl.load(f)
    loc_dict = pkl.load(f)

print(f'Remapped data loaded from {review_pth}')
print(f'#Interaction {len(review_df)}, #User: {n_user}, #POI: {n_poi}')
print(f'Avg.#visit: {len(review_df) / n_user}, density: {len(review_df) / n_user / n_poi}')
print(f'Full data size: {review_df.shape[0]}, #User: {n_user}, #POI: {n_poi}')
print(f'Train size: {trn_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {tst_df.shape[0]}')


print('Generating UI graph...')

ui_nei_dict, ui_edges = gen_nei_graph(review_df, n_user, n_poi)
with open(join(dist_pth, 'ui_graph.pkl'), 'wb') as f:
    pkl.dump(ui_nei_dict, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(ui_edges, f, pkl.HIGHEST_PROTOCOL)

dist_threshold = 1.
print(f'UI graph dumped, generating location graph with delta d: {dist_threshold}km...\n')

dist_mat = None
# if os.path.exists(join(dist_pth, 'dist_mat.npy')):
#     dist_mat = np.load(join(dist_pth, 'dist_mat.npy'))

dist_mat, dist_edges, dist_dict = gen_loc_graph(loc_dict, n_poi, dist_threshold, dist_mat)
with open(join(dist_pth, 'dist_graph.pkl'), 'wb') as f:
    pkl.dump(dist_edges, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(dist_dict, f, pkl.HIGHEST_PROTOCOL)
np.save(join(dist_pth, 'dist_mat.npy'), dist_mat)

dist_on_graph = dist_mat[dist_edges[0], dist_edges[1]]
np.save(join(dist_pth, 'dist_on_graph.npy'), dist_on_graph)
print('Distance graph dumped, process done.')

