import torch
# from model import RNNModel, BasicLSTM
import model
import argparse
import numpy as np
import time


torch_model_dir = "./torch_model_dir/"

# rnn_type = 'LSTM'
ntokens = 10000
nhid = 200
# nlayers = 2
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

def save_torch_ptb_lm():
    RNNModel = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    x = np.arange(80*5*32/20).reshape(20, 32).astype('int64')
    init_hidden = np.zeros((20, 32, 200), dtype='float32')
    init_hidden = torch.tensor(init_hidden)
    init_cell = np.zeros((20, 32, 200), dtype='float32')
    pre_cell = torch.tensor(init_cell)
    x = torch.LongTensor(x)

    RNNModel = torch.jit.trace(RNNModel, [x, init_hidden, pre_cell])
    RNNModel.save(torch_model_dir + "RNNModel.pth")


def test_torch_lm():
    # device = torch.device('cuda:4')
    net = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    # net = loaded_model(pretrained=False)


    x = np.arange(80*5*32/20).reshape(20, 32).astype('int64')
    init_hidden = np.zeros((20, 32, 200), dtype='float32')
    init_hidden = torch.tensor(init_hidden)
    init_cell = np.zeros((20, 32, 200), dtype='float32')
    pre_cell = torch.tensor(init_cell)
    # x = torch.LongTensor(x)

    # net.to(device)
    # x = x.to(device)
    # init_hidden = init_hidden.to(device)
    # init_cell = pre_cell.to(device)

    with torch.no_grad():
        # warmup
        out = net(x, init_hidden, pre_cell)

        t1 = time.time()
        for i in range(100):
            out = net(x, init_hidden, pre_cell)
        t2 = time.time()
        print('torch cost: {} ms.'.format( (t2-t1) * 10))
    

if __name__ == "__main__":
    save_torch_ptb_lm()
    x = np.arange(80*5*32/20).reshape(20, 32).astype('int64')
    init_hidden = np.zeros((20, 32, 200), dtype='float32')
    init_hidden = torch.tensor(init_hidden)
    init_cell = np.zeros((20, 32, 200), dtype='float32')
    pre_cell = torch.tensor(init_cell)
    x = torch.LongTensor(x)

    loaded_model = torch.jit.load(torch_model_dir + "RNNModel.pth")
    _,_,_ = loaded_model(x,init_hidden,pre_cell)

    # test_torch_lm()

    