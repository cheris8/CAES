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
from sklearn.metrics import f1_score, accuracy_score
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
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm','lstm_attn', 'lstm_attn_gy'],
                        required=True,
                        help="Choose model")
    parser.add_argument('--benchmark', type=str, choices=['SCOTUS', 'MSD', 'MSD-Three', 'AAD'],
                        help="Choose benchmark")
    parser.add_argument('--chunk_type', type=str, choices=['length', 'number'],
                        help="Choose chunk type")
    parser.add_argument('--chunk_length', type=int, default=512, 
                        help="Length of each chunk")
    parser.add_argument('--chunk_number', type=int, default=200, 
                        help="The number of chunks")
    parser.add_argument('--metric', type=str, default='all', help="Choose Metric")

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


def obtain_signal(model, iterator, criterion):
    model.eval()
    pred_list = []
    attention_score_list = []
    cnt = 0
    with torch.no_grad():
        for batch in iterator:
            text, label = batch
            text = text.permute(1, 0)
            pred = model(text.long().cuda())
            attention_score = model.attention_score
            attention_score = attention_score.permute(2, 0, 1).to('cpu')
            pred = pred.squeeze(1)
            for att_sc in attention_score:
                attention_score_list.append(att_sc)
            for pr in pred:
                pred_list.append(pr)

    return pred_list, attention_score_list


def calculate_attention(index_per_document, attention_score,):
    att_score_per_doc = {}
    for idx in tqdm(index_per_document.keys()):
        index_of_documents = index_per_document[idx]

        attention_score_per_idx = []
        for x in index_of_documents:
            attention_score_of_idx = attention_score[x]
            attention_score_of_idx = attention_score_of_idx.squeeze(1)
            attention_score_per_idx.append(sum(attention_score_of_idx.tolist()))
        attention_score_per_idx = np.array(attention_score_per_idx).reshape(-1, 1)
        scaler = MinMaxScaler()
        attention_score_per_idx = scaler.fit_transform(attention_score_per_idx)
        att_score_per_doc[idx] = attention_score_per_idx

    return att_score_per_doc

def calculate_confidence(index_per_document, pred_list):
    confidence_score_per_doc = {}
    for idx in tqdm(index_per_document.keys()):
        index_of_documents = index_per_document[idx]

        confidence_per_idx = []
        for x in index_of_documents:
            confidence_of_idx = pred_list[x]
            max_confidence = max(F.softmax(confidence_of_idx, dim=0))
            confidence_per_idx.append(max_confidence.tolist())
        confidence_per_idx = np.array(confidence_per_idx).reshape(-1, 1)
        scaler = MinMaxScaler()
        confidence_per_idx = scaler.fit_transform(confidence_per_idx)
        confidence_score_per_doc[idx] = confidence_per_idx

    return confidence_score_per_doc

def calculate_entropy(index_per_document, pred_list):
    entropy_per_doc = {}
    for idx in tqdm(index_per_document.keys()):
        index_of_documents = index_per_document[idx]

        confidence_per_idx = []
        for x in index_of_documents:
            confidence_of_idx = pred_list[x]
            confidence = F.softmax(confidence_of_idx, dim=0)
            entropy_of_idx = entropy(confidence.to('cpu'))
            ic(1/entropy_of_idx.tolist())
            confidence_per_idx.append(1/entropy_of_idx.tolist())
        scaler = MinMaxScaler()
        confidence_per_idx = np.array(confidence_per_idx).reshape(-1, 1)
        confidence_per_idx = scaler.fit_transform(confidence_per_idx)
        entropy_per_doc[idx] = confidence_per_idx

    return entropy_per_doc


def calculate_all_metric_score(dataset, attention_score, pred_list, datatype):
    if datatype == 'train':
        index_list = dataset.train_splitted_index
    elif datatype == 'valid':
        index_list = dataset.valid_splitted_index
    elif datatype == 'test':
        index_list = dataset.test_splitted_index

    index_per_document = {}
    for i in tqdm(range(len(index_list))):
        if index_list[i] not in index_per_document.keys():
            index_per_document[index_list[i]] = [i]
        else:
            index_per_document[index_list[i]].append(i)

    att_score_per_doc = calculate_attention(index_per_document, attention_score)
    confidence_score_per_doc = calculate_confidence(index_per_document, pred_list)
    entropy_per_doc = calculate_entropy(index_per_document, pred_list)

    return index_per_document, att_score_per_doc, confidence_score_per_doc, entropy_per_doc


def rank_with_all_metrics(dataset, attention_score, pred_list, datatype, n, batch_size, metric, index_per_document, att_score_per_doc, confidence_score_per_doc, entropy_per_doc):
    top_n_chunk_idx_per_doc = {}
    for idx in tqdm(index_per_document.keys()):
        index_of_documents = index_per_document[idx]

        if metric == 'attention':
            final_score = att_score_per_doc[idx]
        elif metric == 'confidence':
            final_score = confidence_score_per_doc[idx]
        elif metric == 'entropy':
            final_score = entropy_per_doc[idx]
        elif metric == 'attention&confidence':
            final_score = att_score_per_doc[idx] + confidence_score_per_doc[idx]
        elif metric == 'attention&entropy':
            final_score = att_score_per_doc[idx] + entropy_per_doc[idx]
        elif metric == 'confidence&entropy':
            final_score = confidence_score_per_doc[idx] + entropy_per_doc[idx]
        elif metric == 'all':
            final_score = att_score_per_doc[idx] + confidence_score_per_doc[idx] + entropy_per_doc[idx]
        
        # TODO 
        rank_result = rank_simple(final_score) 
        ic(len(rank_result))

        for rank in range(n): # rank = topk ? ... n은 topk 개수
            index_of_rank = rank_result.index(rank) # 높은 순서대로 인덱스 가져옴
            document = index_of_documents[index_of_rank]
            if idx not in top_n_chunk_idx_per_doc.keys(): # top n chunk 
                top_n_chunk_idx_per_doc[idx] = [document]
            else:
                top_n_chunk_idx_per_doc[idx].append(document)
        ic(len(top_n_chunk_idx_per_doc))
    
    return concat_chunks(dataset, top_n_chunk_idx_per_doc, datatype, batch_size)


def concat_chunks(dataset, top_n_chunk_idx, datatype, batch_size):
    if datatype == 'train':
        text = dataset.train_text
        label = dataset.train_splitted_label
    elif datatype == 'valid':
        text = dataset.valid_text
        label = dataset.valid_splitted_label
    elif datatype == 'test':
        text = dataset.test_text
        label = dataset.test_splitted_label

    chunk_text_list = []
    chunk_label_list = []
    for doc in top_n_chunk_idx.keys():
        chunk_list = []
        for idx in top_n_chunk_idx[doc]:
            chunk_list.extend(text[idx])
            label_of_chunk = label[idx]
        chunk_text_list.append(chunk_list)
        chunk_label_list.append(label_of_chunk)

    chunk_text = []
    for chunk in chunk_text_list:
        temp_list = []
        for vector in chunk:
            if int(vector) != 1:
                temp_list.append(vector)
        chunk_text.append(temp_list)

    max_text_len = 0
    for chunk in chunk_text:
        if max_text_len < len(chunk):
            max_text_len = len(chunk)

    padded_chunk_text = []
    for i in range(len(chunk_text)):
        padded_chunk_text.append(dataset.pad_tensor(chunk_text[i], max_text_len, 1))
    
    tensor_dataset = TensorDataset(torch.tensor(padded_chunk_text), torch.tensor(chunk_label_list))
    chunk_data_loader = DataLoader(tensor_dataset, batch_size=batch_size)

    return chunk_data_loader


def rank_simple(vector):
    ranked_list = np.array(sorted(range(len(vector)), key=vector.__getitem__))
    reversed_ranked_list = len(ranked_list) - ranked_list.astype(int) - 1
    return reversed_ranked_list.tolist()


def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    dataset = LoadSplittedDataset(
        f'/home/chaehyeong/CAES/data/{args.benchmark}/', 
        args.benchmark, args.batch_size, args.chunk_type, args.chunk_length, args.chunk_number
        )

    # TODO : 
    if args.benchmark == 'SCOTUS':
        PATH = '/home/chaehyeong/CAES/long_text/checkpoints/SCOTUS_lstm_attn_gy_adam_bs1_hd300_acc0.156_model.pth'
    elif args.benchmark == 'MSD':
        PATH = '/home/chaehyeong/CAES/long_text/checkpoints/MSD_lstm_attn_gy_adam_bs8_hd300_acc0.570_model.pth'
    elif args.benchmark == 'MSD-Three':
        PATH = '/home/chaehyeong/CAES/long_text/checkpoints/MSD-Three_lstm_attn_gy_adam_bs8_hd300_acc0.580_model.pth'
    elif args.benchmark == 'AAD':
        PATH = ''
    #PATH = f'/home/chaehyeong/CAES/long_text/checkpoints/{args.benchmark}_lstm_attn_gy_adam_bs8_hd300_acc0.570_model.pth'
    model = torch.load(PATH)
    
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    train_pred, train_attention_score = obtain_signal(model, dataset.dataloader['train'], criterion)
    valid_pred, valid_attention_score = obtain_signal(model, dataset.dataloader['valid'], criterion)
    test_pred, test_attention_score = obtain_signal(model, dataset.dataloader['test'], criterion)

    metric_list = ['attention', 'confidence', 'entropy', 'attention&confidence', 'attention&entropy', 'confidence&entropy', 'all']
    train_index_per_document, train_att_score_per_doc, train_confidence_score_per_doc, train_entropy_per_doc = calculate_all_metric_score(dataset, train_attention_score, train_pred, 'train')
    valid_index_per_document, valid_att_score_per_doc, valid_confidence_score_per_doc, valid_entropy_per_doc = calculate_all_metric_score(dataset, valid_attention_score, valid_pred, 'valid')
    test_index_per_document, test_att_score_per_doc, test_confidence_score_per_doc, test_entropy_per_doc = calculate_all_metric_score(dataset, test_attention_score, test_pred, 'test')
    for metric in metric_list:
        train_chunk_data_loader = rank_with_all_metrics(
            dataset, train_attention_score, train_pred, 'train', 40, args.batch_size, metric, 
            train_index_per_document, train_att_score_per_doc, train_confidence_score_per_doc, train_entropy_per_doc
            )
        valid_chunk_data_loader = rank_with_all_metrics(
            dataset, valid_attention_score, valid_pred, 'valid', 40, args.batch_size, metric, 
            valid_index_per_document, valid_att_score_per_doc, valid_confidence_score_per_doc, valid_entropy_per_doc
            )
        test_chunk_data_loader = rank_with_all_metrics(
            dataset, test_attention_score, test_pred, 'test', 40, args.batch_size, metric, 
            test_index_per_document, test_att_score_per_doc, test_confidence_score_per_doc, test_entropy_per_doc
            )
        
        if args.chunk_type == 'length':
            train_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{metric}_train_chunk_data_loader.pickle'
            valid_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{metric}_valid_chunk_data_loader.pickle'
            test_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_length}_{metric}_test_chunk_data_loader.pickle'
        else:           
            train_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{metric}_train_chunk_data_loader.pickle'
            valid_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{metric}_valid_chunk_data_loader.pickle'
            test_path = f'/home/chaehyeong/CAES/long_text/pickle/{args.benchmark}_{args.chunk_type}_{args.chunk_number}_{metric}_test_chunk_data_loader.pickle'
        with open(train_path, 'wb') as f:
                pickle.dump(train_chunk_data_loader, f, pickle.HIGHEST_PROTOCOL)
        with open(valid_path, 'wb') as f:
                pickle.dump(valid_chunk_data_loader, f, pickle.HIGHEST_PROTOCOL)
        with open(test_path, 'wb') as f:
                pickle.dump(test_chunk_data_loader, f, pickle.HIGHEST_PROTOCOL)

    print("Saved chunks as pickle file!")

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)