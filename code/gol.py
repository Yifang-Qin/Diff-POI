import torch, os, logging, random
import numpy as np
from parse import parse_args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pLog(s: str):
    logging.info(s)

CORES = 16
os.environ['NUMEXPR_MAX_THREADS'] = '16' # mute warnings of logger
DATA_PATH = '../data/processed'
FILE_PATH = './checkpoints/'


ARG = parse_args()
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"
if ARG.log is not None:
    logging.basicConfig(filename=ARG.log, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
else:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

SAVE = ARG.save
LOAD = ARG.load
SEED = ARG.seed
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
EPOCH = ARG.epoch
PATH = ARG.path
dataset = ARG.dataset
patience = ARG.patience

seed_torch(SEED)
os.makedirs(FILE_PATH, exist_ok=True)

dist_mat = torch.from_numpy(np.load(os.path.join(DATA_PATH, dataset.lower(), 'dist_mat.npy')))
device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
conf = {'lr': ARG.lr, 'decay': ARG.decay, 'num_layer': ARG.layer, 'hidden': ARG.hidden,
        'dropout': ARG.dropout, 'keepprob': ARG.keepprob, 'max_len': ARG.length,
        'interval': ARG.interval, 'T': ARG.diffsize, 'beta': ARG.beta, 'dt': ARG.stepsize}
