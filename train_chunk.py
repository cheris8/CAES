import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from tensorboardX import SummaryWriter
from glob import glob

from torch.utils.data.dataset import TensorDataset

from dataset import LoadDataset, LoadSplittedDataset
from model import RNN, LSTM, CNN, LSTM_with_Attention, ATT_LSTM, SELF_ATTEN_VEC

from icecream import ic
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import entropy

from sklearn.preprocessing import MinMaxScaler
import pickle5 as pickle

def generate_experiment_name():
    dirs = glob('runs/*')
    indices = [int(val[-3:]) for val in dirs if val[-3:].isdigit()]
    if indices:
        last_idx = max(max(indices), 0)
    else:
        last_idx = 0

    return f'runs/experiments{last_idx+1:03d}'


writer = SummaryWriter(generate_experiment_name())
torch.manual_seed(252)
torch.cuda.manual_seed(252)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Project", add_help=False)
    parser.add_argument('--optim', type=str, choices=['sgd', 'adam'],
                        default='sgd', help="Choose optimizer")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--epoch', type=int, default=15,
                        help="Epoch")
    parser.add_argument('--cuda', action='store_true',
                        help="Use cuda")
    parser.add_argument('--ed', type=int, default=300,
                        help="Embedding dimensions")
    parser.add_argument('--dropout', type=float, default=0.7,
                        help="Drop out")
    parser.add_argument('--word_vector', type=str2bool, nargs='?',
                        default=True, help="Use word vector like gloVe")
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'lstm_attn', 'lstm_attn_gy'],
                        required=True,
                        help="Choose model")
    parser.add_argument('--benchmark', type=str, choices=['SCOTUS', 'MSD', 'MSD-Three', 'AAD'],
                        help="Choose benchmark")
    parser.add_argument('--chunk_type', type=str, default='number', choices=['number', 'length'],
                        help="Choose chunking type")
    parser.add_argument('--chunk_length', type=int, default=512,
                        help="Length of each chunk")
    parser.add_argument('--chunk_number', type=int, default=200,
                        help="The number of chunks")
    parser.add_argument('--metric', type=str, default='all', 
                        help="Choose Metric")
    parser.add_argument('--rootpath', type=str, default='/home/minju/CAES/')

    parser_args, _ = parser.parse_known_args()

    target_parser = argparse.ArgumentParser(parents=[parser])
    
    if parser_args.model in 'rnn':
        target_parser.add_argument('--hd', type=int, default=512,
                                   help="Hidden dimensions")
    elif parser_args.model in ['lstm', 'lstm_attn', 'lstm_attn_gy']:
        target_parser.add_argument('--hd', type=int, default=512,
                                   help="Hidden dimensions")
        target_parser.add_argument('--layer', type=int, default=2,
                                   help="Layer number")
        target_parser.add_argument('--bidirectional', action = 'store_true',
                                   default=True, help="use bidirectional")

    args = target_parser.parse_args()
    writer.add_text('Hyperparameters', str(args))

    return args


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    # convert into float for division
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def accuracy(preds, y):
    preds = preds.cpu()
    y = y.cpu()
    argmax_preds = torch.argmax(preds, dim=1)
    return accuracy_score(y, argmax_preds)

def precision(preds, y):
    preds = preds.cpu()
    y = y.cpu()
    argmax_preds = torch.argmax(preds, dim=1)
    return precision_score(y, argmax_preds, average='macro')

def recall(preds, y):
    preds = preds.cpu()
    y = y.cpu()
    argmax_preds = torch.argmax(preds, dim=1)
    return recall_score(y, argmax_preds, average='macro')

def f1score(preds, y):
    preds = preds.cpu()
    y = y.cpu()
    argmax_preds = torch.argmax(preds, dim=1)
    return f1_score(y, argmax_preds, average='macro')

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    dataset = LoadSplittedDataset(
        f'/home/chaehyeong/CAES/data/{args.benchmark}/', 
        args.benchmark, args.batch_size, args.chunk_type, args.chunk_length, args.chunk_number
        )

    # Hyper parameters
    INPUT_DIM = len(dataset.vocab)
    if args.benchmark == 'SCOTUS':
        OUTPUT_DIM = 14
    elif args.benchmark == 'MSD':
        OUTPUT_DIM = 4
    elif args.benchmark == 'MSD-Three':
        OUTPUT_DIM = 3
    else:
        OUTPUT_DIM = 4

    if args.model == 'rnn':
        print("Model: Vanila RNN")
        model = RNN(INPUT_DIM, args.ed, args.hd, OUTPUT_DIM).to(device)
        chunk_model = RNN(INPUT_DIM, args.ed, args.hd, OUTPUT_DIM).to(device)
    elif args.model == 'lstm':
        print("Model: LSTM")
        model = LSTM(
            INPUT_DIM, args.ed, args.hd, OUTPUT_DIM,
            n_layers=args.layer, use_bidirectional=args.bidirectional,
            use_dropout=args.dropout).to(device)
    elif args.model == 'lstm_attn':
        print("Model: LSTM with Attension")
        model = LSTM_with_Attention(
            INPUT_DIM, args.ed, args.hd, OUTPUT_DIM,
            n_layers=args.layer, use_bidirectional=args.bidirectional,
            dropout=args.dropout).to(device)
    elif args.model == 'lstm_attn_gy':
        print("Model: Gayeon's LSTM with attention")
        model = SELF_ATTEN_VEC(
            INPUT_DIM, args.ed, args.hd, OUTPUT_DIM,
            n_layers=args.layer, use_bidirectional=bool(args.bidirectional),
            dropout=args.dropout).to(device)

    if args.optim == 'sgd':
        print("Optim: SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        print("Optim: Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    best_acc = 0
    for epoch in range(args.epoch):

        if args.chunk_type == 'length':
            train_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{args.metric}_train_chunk_data_loader.pickle'
            valid_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{args.metric}_valid_chunk_data_loader.pickle'
            test_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{args.metric}_test_chunk_data_loader.pickle'
        else:
            train_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{args.metric}_train_chunk_data_loader.pickle'
            valid_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{args.metric}_valid_chunk_data_loader.pickle'
            test_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{args.metric}_test_chunk_data_loader.pickle'            
        with open(train_path, 'rb') as f:
            train_chunk_data_loader = pickle.load(f)
        with open(valid_path, 'rb') as f:
            valid_chunk_data_loader = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_chunk_data_loader = pickle.load(f)

        train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_chunk_data_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = evaluate(model, valid_chunk_data_loader, criterion)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_chunk_data_loader, criterion)
        print(f'Epoch: {epoch+1:02}, \
            Train Loss: {train_loss:.3f}, Train Acc: {train_acc * 100:.2f}%, Train F1: {train_f1 * 100:.2f}%, \
            Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc * 100:.2f}%, Val. F1: {valid_f1 * 100:.2f}%, \
            Test Loss: {test_loss:.3f}, Test Acc: {test_acc * 100:.2f}%, Test F1: {test_f1 * 100:.2f}%')

        writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': valid_loss,
            }, epoch + 1)
        writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': valid_acc,
            }, epoch + 1)

        if best_acc <= valid_acc:
            best_acc = valid_acc
            acc_result = valid_acc
            if args.chunk_type == 'length':
                modelname = "/home/chaehyeong/CAES/long_text/checkpoints/{}_{}_{}_{}_{}_bs{}_hd{}_acc{:.03f}_model.pth".format(
                    args.benchmark, args.chunk_type, args.chunk_length, args.metric, args.model, args.batch_size, args.hd, valid_acc
                    )
            else:
                modelname = "/home/chaehyeong/CAES/long_text/checkpoints/{}_{}_{}_{}_{}_bs{}_hd{}_acc{:.03f}_model.pth".format(
                    args.benchmark, args.chunk_type, args.chunk_number, args.metric, args.model, args.batch_size, args.hd, valid_acc
                    )
            #modelname = "/home/chaehyeong/CAES/long_text/checkpoints/{}_{}_{}_{}_{}_bs{}_hd{}_acc{:.03f}_model.pth".format(
            #    args.benchmark, args.chunk_type, args.metric, args.model, args.optim, args.batch_size, args.hd, valid_acc
            #    )
            torch.save(model, modelname)
    writer.add_text('Test acc', str(acc_result))
    torch.cuda.empty_cache()


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_f1 = 0

    model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        text, label = batch
        text = text.permute(1, 0)
        pred = model(text.long().cuda()).squeeze(1)
        
        loss = criterion(pred, label.long().cuda())
        
        acc = accuracy(pred, label.long().cuda())
        prec = precision(pred, label.long().cuda())
        rec = recall(pred, label.long().cuda())
        f1 = f1score(pred, label.long().cuda())
        # f1 = f1score(pred, label.long())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_rec += rec.item()
        epoch_f1 += f1.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_rec / len(iterator), epoch_f1 / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_f1 = 0

    model.eval()
    for batch in iterator:
        text, label = batch
        text = text.permute(1, 0)
        pred = model(text.long().cuda()).squeeze(1)

        loss = criterion(pred, label.long().cuda())
        acc = accuracy(pred, label.long().cuda())
        prec = precision(prec, label.long().cuda())
        rec = recall(prec, label.long().cuda())
        f1 = f1score(prec, label.long().cuda())
        # f1 = f1score(pred, label.long())

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_rec += rec.item()
        epoch_f1 += f1.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_rec / len(iterator), epoch_f1 / len(iterator)

    
if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)