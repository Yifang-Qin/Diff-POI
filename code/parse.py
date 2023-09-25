import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Graph ODE for recommendation")
    parser.add_argument('--dataset', type=str, default='foursquare', # foursquare denotes singapore dataset
                        help="available datasets: ['foursquare', 'gowalla', 'nyc', 'tky']")
    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--batch', type=int, default=1024,
                        help="the batch size for training procedure")
    parser.add_argument('--testbatch', type=int, default=1024,
                        help="the batch size of users for testing")
    parser.add_argument('--length', type=int, default=100,
                        help="max sequence length")
    parser.add_argument('--beta', type=float, default=0.2,
                        help="fisher loss weight")
    parser.add_argument('--hidden', type=int, default=64,
                        help="node embedding size")
    parser.add_argument('--interval', type=int, default=256,
                        help="types of temporal and locational intervals")
    parser.add_argument('--layer', type=int, default=2,
                        help="layer num of GNN")
    parser.add_argument('--diffsize', type=int, default=1,
                        help="diffusion size T")
    parser.add_argument('--stepsize', type=float, default=0.01,
                        help="diffusion step size dt")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3,
                        help="weight decay for l2 normalizaton")
    parser.add_argument('--dropout', action='store_true', default=False,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="dropout probalitity")
    parser.add_argument('--patience', type=int, default=10,
                        help="early stop patience")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help='path to save weights')
    parser.add_argument('--log', type=str, default=None,
                        help="log file path")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed')
    parser.add_argument('--gpu', type=str, default=None,
                        help='training device')
    return parser.parse_args()
