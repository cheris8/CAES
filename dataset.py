import os

import torch
from torchtext.legacy.data import Dataset

from icecream import ic

import nltk
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer
import random
import pandas as pd
import copy

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
import spacy
from collections import Counter
from spacy.lang.en import English
import pickle5 as pickle
from tqdm import tqdm


torch.manual_seed(252)
torch.cuda.manual_seed(252)
random.seed(252)

class SimpleDatset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label
    def __getitem__(self, item):
        text_item = self.text[item]
        label_item = self.label[item]
        return text_item, label_item
    def __len__(self):
        return len(self.text)

class LoadDataset(Dataset):
    def __init__(self, args, file_path, benchmark, batch_size):
        self.file_path = file_path
        self.benchmark = benchmark
        self.batch_size = batch_size

        # ic(torch.__version__)
        # ic(torch.cuda.is_available())

        self.tokenizer = get_tokenizer('basic_english')
        # self.tokenizer = get_tokenizer('spacy', language='en')
        # self.tokenizer = spacy.load("en_core_web_sm")

        print("Loading files...")

        if 'data' in self.file_path:
            self.train_data = pd.read_csv(file_path + 'train.csv')
            self.valid_data = pd.read_csv(file_path + 'valid.csv')
            self.test_data = pd.read_csv(file_path + 'test.csv')
        elif 'tsv' in file_path:
            self.data = pd.read_csv(self.file_path, delimiter='\t')
        else:
            print("Wrong data type")
        
        self.train_data['text'] = self.train_data['text'].astype(str)
        self.valid_data['text'] = self.valid_data['text'].astype(str)
        self.test_data['text'] = self.test_data['text'].astype(str)
        self.train_rawtext = list(self.train_data['text'])
        self.valid_rawtext = list(self.valid_data['text'])
        self.test_rawtext = list(self.test_data['text'])
    

        if self.benchmark == 'SCOTUS':
            self.train_label = list(self.train_data['label'])
            self.valid_label = list(self.valid_data['label'])
            self.test_label = list(self.test_data['label'])
        elif self.benchmark == 'MSD':
            self.train_label_str = list(self.train_data['rating'])
            self.valid_label_str = list(self.valid_data['rating'])
            self.test_label_str = list(self.test_data['rating'])
            self.label_to_id = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])
        elif self.benchmark == 'MSD-Three':
            self.train_label_str = list(self.train_data['rating'])
            self.valid_label_str = list(self.valid_data['rating'])
            self.test_label_str = list(self.test_data['rating'])
            self.label_to_id = {'LOW': 0, 'MED': 1, 'HIGH': 2}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])        
        else:
            self.train_label_str = list(self.train_data['subject'])
            self.valid_label_str = list(self.valid_data['subject'])
            self.test_label_str = list(self.test_data['subject'])
            # self.label_to_id = {'cs.AI': 0, 'cs.CE': 1, 'cs.cv': 2, 'cs.DS': 3, 'cs.IT': 4, 'cs.NE': 5, 'cs.PL': 6, 'cs.SY': 7, 'math.AC': 8, 'math.GR': 9, 'math.ST': 10}
            self.label_to_id = {'cs.IT': 0, 'cs.NE': 1, 'math.AC': 2, 'math.GR': 3}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])

        print("Tokenizing...")

        self.train_tokenized_text = []
        self.valid_tokenized_text = []
        self.test_tokenized_text = []
        for text in tqdm(self.train_rawtext):
            self.train_tokenized_text.append(self.tokenizer(text))
        for text in tqdm(self.valid_rawtext):
            self.valid_tokenized_text.append(self.tokenizer(text))
        for text in tqdm(self.test_rawtext):
            self.test_tokenized_text.append(self.tokenizer(text))

        self.max_text_len = 0
        for i in range(len(self.train_tokenized_text)):
            if self.max_text_len < len(self.train_tokenized_text[i]):
                self.max_text_len = len(self.train_tokenized_text[i])
        for i in range(len(self.valid_tokenized_text)):
            if self.max_text_len < len(self.valid_tokenized_text[i]):
                self.max_text_len = len(self.valid_tokenized_text[i])
        for i in range(len(self.test_tokenized_text)):
            if self.max_text_len < len(self.test_tokenized_text[i]):
                self.max_text_len = len(self.test_tokenized_text[i])

        self.all_tokenized_text = []
        self.all_tokenized_text.extend(self.train_tokenized_text)
        self.all_tokenized_text.extend(self.valid_tokenized_text)
        self.all_tokenized_text.extend(self.test_tokenized_text)

        print("Building vocab...")

        vocab_path = args.rootpath + f'pickle/vocab_{benchmark}.pickle'
        if os.path.isfile(vocab_path):
            print("Vocab is already built!")
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
                self.vocab_stoi = vars(self.vocab)['stoi']
                self.vocab_size = len(self.vocab)

        else:
            self.vocab = build_vocab_from_iterator(self.all_tokenized_text)
            self.vocab_stoi = vars(self.vocab)['stoi']
            self.vocab_size = len(self.vocab)

            with open(vocab_path, 'wb') as f:
                pickle.dump(self.vocab, f, pickle.HIGHEST_PROTOCOL)
        
        print("Encoding...")

        self.train_tokenized_vecs = []
        self.valid_tokenized_vecs = []
        self.test_tokenized_vecs = []
        for text in tqdm(self.train_tokenized_text):
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.train_tokenized_vecs.append(vec_list)
        for text in tqdm(self.valid_tokenized_text):
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.valid_tokenized_vecs.append(vec_list)
        for text in tqdm(self.test_tokenized_text):
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.test_tokenized_vecs.append(vec_list)

        print("Padding...")

        # self.train_text = []
        # self.valid_text = []
        # self.test_text = []
        # for i in tqdm(range(len(self.train_tokenized_vecs))):
        #     self.train_text.append(self.pad_tensor(self.train_tokenized_vecs[i], self.max_text_len, 1))
        # for i in tqdm(range(len(self.valid_tokenized_vecs))):
        #     self.valid_text.append(self.pad_tensor(self.valid_tokenized_vecs[i], self.max_text_len, 1))
        # for i in tqdm(range(len(self.test_tokenized_vecs))):
        #     self.test_text.append(self.pad_tensor(self.test_tokenized_vecs[i], self.max_text_len, 1))

        # self.tensor_dataset = {}
        # self.tensor_dataset['train'] = TensorDataset(torch.tensor(self.train_text), torch.tensor(self.train_label))
        # self.tensor_dataset['valid'] = TensorDataset(torch.tensor(self.valid_text), torch.tensor(self.valid_label))
        # self.tensor_dataset['test'] = TensorDataset(torch.tensor(self.test_text), torch.tensor(self.test_label))
        
        # self.dataloader = {}
        # self.dataloader['train'] = DataLoader(self.tensor_dataset['train'], batch_size=self.batch_size)
        # self.dataloader['valid'] = DataLoader(self.tensor_dataset['valid'], batch_size=self.batch_size)
        # self.dataloader['test'] = DataLoader(self.tensor_dataset['test'], batch_size=self.batch_size)

        # self.train_text = SimpleDatset(torch.Tensor(self.train_tokenized_vecs), torch.Tensor(self.train_label))
        # self.valid_text = SimpleDatset(torch.Tensor(self.valid_tokenized_vecs), torch.Tensor(self.train_label))
        # self.test_text = SimpleDatset(torch.Tensor(self.test_tokenized_vecs), torch.Tensor(self.train_label))

        self.train_text = SimpleDatset(self.train_tokenized_vecs, self.train_label)
        self.valid_text = SimpleDatset(self.valid_tokenized_vecs, self.train_label)
        self.test_text = SimpleDatset(self.test_tokenized_vecs, self.train_label)

        self.dataloader = {}
        self.dataloader['train'] = DataLoader(self.train_text, batch_size=self.batch_size, collate_fn=self.collate_fn_padd)
        self.dataloader['valid'] = DataLoader(self.valid_text, batch_size=self.batch_size, collate_fn=self.collate_fn_padd)
        self.dataloader['test'] = DataLoader(self.test_text, batch_size=self.batch_size, collate_fn=self.collate_fn_padd)

        print("Dataset Loaded!")

    def collate_fn_padd(self, batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        text, label = zip(*batch)
        device = torch.device("cpu")

        text_batch = []
        lengths = torch.tensor([len(t) for t in text]).to(device)
        max_len = max(lengths)
        for t in text:
            text_batch.append(self.pad_tensor(t, max_len, 1))

        # mask = (text_batch != 1).to(device)
        # batch = SimpleDatset(torch.Tensor(text_batch), torch.Tensor(label))

        return torch.Tensor(text_batch), torch.Tensor(label)

    def yield_tokens(self, text_list):
        for text in text_list:
            yield self.tokenizer(text)
            
    def pad_tensor(self, vec, pad_len, dim, _pad=1):

        pad_size = pad_len - len(vec)
        vec = torch.tensor(vec).unsqueeze(0)

        if _pad == 0:
            padding_list = torch.zeros(pad_size)
        else:
            padding_list = torch.ones(pad_size).unsqueeze(0)

        return torch.cat([vec, padding_list], dim=dim).squeeze(0).tolist()

    def build_vocab(self, texts, max_vocab=25000, min_freq=3):
        wc = Counter()
        for doc in list(self.tokenizer.pipe(texts)):
            for word in doc:
                wc[word.lower_] += 1

        word2id = {}
        id2word = {}
        for word, count in wc.most_common():
            if count < min_freq: break
            if len(word2id) >= max_vocab: break
            wid = len(word2id)
            word2id[word] = wid
            id2word[wid] = word
        return word2id, id2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

class LoadSplittedDataset(Dataset):
    def __init__(self, file_path, benchmark, batch_size, chunk_type, chunk_length, chunk_number):
        self.file_path = file_path
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.chunk_type = chunk_type
        self.chunk_length = chunk_length
        self.chunk_number = chunk_number

        self.tokenizer = get_tokenizer('basic_english')
        # self.tokenizer = get_tokenizer('spacy', language='en')
        # self.tokenizer = spacy.load("en_core_web_sm")

        print("Loading files...")

        if 'data' in self.file_path:
            self.train_data = pd.read_csv(file_path + 'train.csv')
            self.valid_data = pd.read_csv(file_path + 'valid.csv')
            self.test_data = pd.read_csv(file_path + 'test.csv')
        else:
            print("Wrong data type")
        self.train_data['text'] = self.train_data['text'].astype(str)
        self.valid_data['text'] = self.valid_data['text'].astype(str)
        self.test_data['text'] = self.test_data['text'].astype(str)
        self.train_rawtext = list(self.train_data['text'])
        self.valid_rawtext = list(self.valid_data['text'])
        self.test_rawtext = list(self.test_data['text'])

        if self.benchmark == 'SCOTUS':
            self.train_label = list(self.train_data['label'])
            self.valid_label = list(self.valid_data['label'])
            self.test_label = list(self.test_data['label'])
        elif self.benchmark == 'MSD':
            self.train_label_str = list(self.train_data['rating'])
            self.valid_label_str = list(self.valid_data['rating'])
            self.test_label_str = list(self.test_data['rating'])
            self.label_to_id = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])          
        elif self.benchmark == 'MSD-Three':
            self.train_label_str = list(self.train_data['rating'])
            self.valid_label_str = list(self.valid_data['rating'])
            self.test_label_str = list(self.test_data['rating'])
            self.label_to_id = {'LOW': 0, 'MED': 1, 'HIGH': 2}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])
        else:
            self.train_label_str = list(self.train_data['subject'])
            self.valid_label_str = list(self.valid_data['subject'])
            self.test_label_str = list(self.test_data['subject'])
            # self.label_to_id = {'cs.AI': 0, 'cs.CE': 1, 'cs.CV': 2, 'cs.DS': 3, 'cs.IT': 4, 'cs.NE': 5, 'cs.PL': 6, 'cs.SY': 7, 'math.AC': 8, 'math.GR': 9, 'math.ST': 10}
            self.label_to_id = {'cs.IT': 0, 'cs.NE': 1, 'math.AC': 2, 'math.GR': 3}
            self.train_label = []
            self.valid_label = []
            self.test_label = []
            for l in self.train_label_str:
                self.train_label.append(self.label_to_id[l])
            for l in self.valid_label_str:
                self.valid_label.append(self.label_to_id[l])
            for l in self.test_label_str:
                self.test_label.append(self.label_to_id[l])

        print("Tokenizing...")

        self.train_tokenized_text = []
        self.valid_tokenized_text = []
        self.test_tokenized_text = []
        for text in self.train_rawtext:
            self.train_tokenized_text.append(self.tokenizer(text))
        for text in self.valid_rawtext:
            self.valid_tokenized_text.append(self.tokenizer(text))
        for text in self.test_rawtext:
            self.test_tokenized_text.append(self.tokenizer(text))

        self.max_text_len = 0
        for i in range(len(self.train_tokenized_text)):
            if self.max_text_len <= len(self.train_tokenized_text[i]):
                self.max_text_len = len(self.train_tokenized_text[i])
        for i in range(len(self.valid_tokenized_text)):
            if self.max_text_len <= len(self.valid_tokenized_text[i]):
                self.max_text_len = len(self.valid_tokenized_text[i])
        for i in range(len(self.test_tokenized_text)):
            if self.max_text_len <= len(self.test_tokenized_text[i]):
                self.max_text_len = len(self.test_tokenized_text[i])

        self.all_tokenized_text = []
        self.all_tokenized_text.extend(self.train_tokenized_text)
        self.all_tokenized_text.extend(self.valid_tokenized_text)
        self.all_tokenized_text.extend(self.test_tokenized_text)

        print("Loading vocab...")

        with open(f'/home/chaehyeong/CAES/long_text/pickle/vocab_{self.benchmark}.pickle', 'rb') as f:
            self.vocab = pickle.load(f)     
        self.vocab_stoi = vars(self.vocab)['stoi']
        # self.vocab_size = len(self.vocab)
        
        print("Encoding...")

        self.train_tokenized_vecs = []
        self.valid_tokenized_vecs = []
        self.test_tokenized_vecs = []
        for text in self.train_tokenized_text:
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.train_tokenized_vecs.append(vec_list)
        for text in self.valid_tokenized_text:
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.valid_tokenized_vecs.append(vec_list)
        for text in self.test_tokenized_text:
            vec_list = []
            for token in text:
                vec_list.append(self.vocab_stoi[token])
            self.test_tokenized_vecs.append(vec_list)
        
        # TODO
        # split_num : 한 chunk의 길이
        if self.chunk_type == 'length':
            split_num = self.chunk_length
        else:
            # split_num = int(len(text)/50) # 50 = chunk 개수
            split_num = int(len(text)/self.chunk_number)
        ic(split_num)

        self.train_splitted_text = []
        self.train_splitted_label = []
        self.train_splitted_index = []
        self.valid_splitted_text = []
        self.valid_splitted_label = []
        self.valid_splitted_index = []
        self.test_splitted_text = []
        self.test_splitted_label = []
        self.test_splitted_index = []

        for i in range(len(self.train_tokenized_vecs)):
            text = self.train_tokenized_vecs[i]
            chunks = [text[x:x+split_num] for x in range(0, len(text), split_num)]
            for x in chunks:
                self.train_splitted_index.append(i)
                self.train_splitted_text.append(x)
                self.train_splitted_label.append(self.train_label[i])
        for i in range(len(self.valid_tokenized_vecs)):
            text = self.valid_tokenized_vecs[i]
            chunks = [text[x:x+split_num] for x in range(0, len(text), split_num)]
            for x in chunks:
                self.valid_splitted_index.append(i)
                self.valid_splitted_text.append(x)
                self.valid_splitted_label.append(self.valid_label[i])
        for i in range(len(self.test_tokenized_vecs)):
            text = self.test_tokenized_vecs[i]
            chunks = [text[x:x+split_num] for x in range(0, len(text), split_num)]
            for x in chunks:
                self.test_splitted_index.append(i)
                self.test_splitted_text.append(x)
                self.test_splitted_label.append(self.test_label[i])

        self.max_text_len = 0
        for i in range(len(self.train_splitted_text)):
            if self.max_text_len < len(self.train_splitted_text[i]):
                self.max_text_len = len(self.train_splitted_text[i])
        for i in range(len(self.valid_splitted_text)):
            if self.max_text_len < len(self.valid_splitted_text[i]):
                self.max_text_len = len(self.valid_splitted_text[i])
        for i in range(len(self.test_splitted_text)):
            if self.max_text_len < len(self.test_splitted_text[i]):
                self.max_text_len = len(self.test_splitted_text[i])
    
        self.train_text = []
        for i in range(len(self.train_splitted_text)):
            self.train_text.append(self.pad_tensor(self.train_splitted_text[i], self.max_text_len, 1))
        self.valid_text = []
        for i in range(len(self.valid_splitted_text)):
            self.valid_text.append(self.pad_tensor(self.valid_splitted_text[i], self.max_text_len, 1))
        self.test_text = []
        for i in range(len(self.test_splitted_text)):
            self.test_text.append(self.pad_tensor(self.train_splitted_text[i], self.max_text_len, 1))
            
        self.tensor_dataset = {}
        self.tensor_dataset['train'] = TensorDataset(torch.tensor(self.train_text), torch.tensor(self.train_splitted_label))
        self.tensor_dataset['valid'] = TensorDataset(torch.tensor(self.valid_text), torch.tensor(self.valid_splitted_label))
        self.tensor_dataset['test'] = TensorDataset(torch.tensor(self.test_text), torch.tensor(self.test_splitted_label))

        self.dataloader = {}
        self.dataloader['train'] = DataLoader(self.tensor_dataset['train'], batch_size=self.batch_size)
        self.dataloader['valid'] = DataLoader(self.tensor_dataset['valid'], batch_size=self.batch_size)
        self.dataloader['test'] = DataLoader(self.tensor_dataset['test'], batch_size=self.batch_size)


    def yield_tokens(self, text_list):
        for text in text_list:
            yield self.tokenizer(text)
            
    def pad_tensor(self, vec, pad_len, dim, _pad=1):
        pad_size = pad_len - len(vec)
        vec = torch.tensor(vec).unsqueeze(0)
        if _pad == 0:
            padding_list = torch.zeros(pad_size)
        else:
            padding_list = torch.ones(pad_size).unsqueeze(0)

        return torch.cat([vec, padding_list], dim=dim).squeeze(0).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
