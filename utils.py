import ast

import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset

import matplotlib.pyplot as plt

import os

import pandas as pd

import pickle

import random

import socket

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data

from tqdm import tqdm

from transformers import BertTokenizer

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_on_server():
    # only works because I use macOS
    if torch.backends.mps.is_available():
        return False
    else:
        return True

def is_connected_to_internet():
    try:
        # Connect to the DNS server at 8.8.8.8 (Google) on port 53 (DNS)
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False
    
class StandardDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
class SingleDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index]
    
class TokenizedStandardDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_length=512):
        super().__init__()
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        input_text = self.x[index]
        encoded_input = self.tokenizer(input_text, return_tensors='pt', padding="max_length", truncation=True, max_length=self.max_length)
        for key in encoded_input.keys():
            encoded_input[key] = encoded_input[key].squeeze()
        return encoded_input, self.y[index]

class TokenizedDataset(Dataset):
    """
    needed by Trainer class from huggingface
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}  # Remove batch dimension
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

########################################################################################################################
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(labels, pred)
    return {"accuracy": accuracy}

########################################################################################################################
def get_train_val_test_split(dataset: PygNodePropPredDataset, size):
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']  # Tensor of training node indices
    val_idx = split_idx['valid']  # Tensor of validation node indices
    test_idx = split_idx['test']  # Tensor of test node indices

    train_mask = torch.zeros(size, dtype=torch.bool)
    train_mask[train_idx] = True

    val_mask = torch.zeros(size, dtype=torch.bool)
    val_mask[val_idx] = True

    test_mask = torch.zeros(size, dtype=torch.bool)
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def encode_texts(texts, model, tokenizer, device):
    encoded_batch = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors='pt').to(device)
    encoded_batch = {key: value for key, value in encoded_batch.items() if key != 'token_type_ids'}
    with torch.no_grad():
        outputs = model(**encoded_batch)
    if "bert-base-uncased" in model.config.name_or_path:
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif model.config.name_or_path == "gpt2":
        hidden_states = outputs[0]
        input_ids = encoded_batch['input_ids']
        batch_size, sequence_length = input_ids.shape
        sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1 # where does padding start
        sequence_lengths = sequence_lengths % input_ids.shape[-1] # cannot larger than input
        sequence_lengths = sequence_lengths.to(hidden_states.device) #  move to right device
        embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths] # last hidden state
    return embeddings.cpu().numpy()

def save_text_in_encoded_form(device, model, tokenizer, texts: list, file_name: str='saved_embeddings/my_model.npy'):
    model.to(device)
    text_data = SingleDataset(texts)
    text_loader = DataLoader(text_data, batch_size=8, shuffle=False)
    model.eval()
    encoded_tests = []
    for text in tqdm(text_loader):
        encoded_tests.append(encode_texts(text, model, tokenizer, device))
    all_embeddings = np.vstack(encoded_tests)
    np.save(file_name, all_embeddings)

def get_bert_embeddings_for_nodes(dataset_name: str):
    if dataset_name == 'ogbn-arxiv':
        dataset_bert = PygNodePropPredDataset(name='ogbn-arxiv')
        data_Bert = dataset_bert[0]
        nodeidx2paperid = pd.read_csv('dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
        df_embeddings = pd.read_csv('dataset/bert_encoded/all_consistent.csv')
        # Merge on 'paper id' to align embeddings with node indices
        merged_df = pd.merge(nodeidx2paperid, df_embeddings, on='paper id', how='left')
        embedding_matrix = merged_df['bert_embeddings'].apply(ast.literal_eval)
        new_x = torch.tensor(embedding_matrix, dtype=torch.float)
        bert_graph_data = Data(x = new_x,
                    edge_index=data_Bert.edge_index,
                    node_year=data_Bert.node_year,
                    y=data_Bert.y)
        return bert_graph_data # TODO change to dataset_bert
    
def add_noise(text, word_list, random_word_probability=0.15, random_drop_probability=0.15):
    words = text.split()
    noisy_text = []
    
    for word in words:
        if random.random() < random_word_probability: # add a random word from the list
            noisy_text.append(random.choice(word_list))
        if random.random() < random_drop_probability: # ignore current word
            continue
        else:
            noisy_text.append(word)
    
    
    return ' '.join(noisy_text)

def get_text_for_nodes(dataset_name: str, only_title=False, noisy=False) -> pd.DataFrame:
    if dataset_name == 'ogbn-arxiv':
        prefix = ""
        if os.path.exists("transfolder # i insert this text so it will ignore this part that was used for testing"):
            prefix = "transfolder/"
        # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        # data = dataset[0]
        nodeidx2paperid = pd.read_csv(prefix + 'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz',
                                        compression='gzip')
        raw_text = pd.read_csv(prefix + 'dataset/ogbn_arxiv_orig/titleabs.tsv',
                                sep='\t', header=None, names=['paper id', 'title', 'abs'])
        df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
        df['node idx'] = df['node idx'].astype(int)  # Convert node idx to int if it's not
        df.set_index('node idx', inplace=True)  # Set node idx as index if not already
        if only_title:
            if noisy:
                df['title'] = df['title'].apply(add_noise)
            df['text'] = df['title']
        else:
            if noisy:
                most_common_words_file_name = "/home/yourfirstname.yourlastname/experiment/dataset/arxiv_most_common_words.pkl" if is_on_server() else "most_common_words.pkl"
                with open(most_common_words_file_name, "rb") as file:  # 'rb' mode is for reading binary files
                    word_list = pickle.load(file)
                df['title'] = df['title'].apply(add_noise, args=(word_list,))
                df['abs'] = df['abs'].apply(add_noise, args=(word_list,))
            df['text'] = df['title'] + " [SEP] " + df['abs']
        return df
    elif dataset_name == 'ogbn-products':
        if is_on_server():
            raw_text = pd.read_csv("/home/yourfirstname.yourlastname/experiment/dataset/Amazon-3M.raw/full_asin.csv")
            nodeid2asin = pd.read_csv("/home/yourfirstname.yourlastname/experiment/dataset/ogbn_products/mapping/nodeidx2asin.csv.gz", compression="gzip")
        else:
            raw_text = pd.read_csv("/Users/yourfirstnameyourlastname/Downloads/Amazon-3M.raw/full_asin.csv")
            nodeid2asin = pd.read_csv("/Users/yourfirstnameyourlastname/Downloads/ogbn_products/mapping/nodeidx2asin.csv.gz", compression="gzip")
        merged_df = pd.merge(nodeid2asin, raw_text, left_on='asin', right_on='uid', how='inner')
        merged_df = merged_df.fillna('')
        if noisy:
                most_common_words_file_name = "/home/yourfirstname.yourlastname/experiment/dataset/most_common_products_words.pkl" if is_on_server() else "most_common_products_words.pkl"
                with open(most_common_words_file_name, "rb") as file:  # 'rb' mode is for reading binary files
                    word_list = pickle.load(file)
                merged_df['title'] = merged_df['title'].apply(add_noise, args=(word_list,))
                merged_df['content'] = merged_df['content'].apply(add_noise, args=(word_list,))
        merged_df['text'] = merged_df['title'] + " [SEP] " + merged_df['content']
        return merged_df
    
def train_just_classifer():
    pass


########################################################################################################################
def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'epoch': epoch + 1,  # epoch + 1 because we want to start from the next epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    else:
        epoch = 1
    return model, optimizer, epoch

def get_file_names_containing_substring(directory, substring='distill'):
    files = os.listdir(directory)
    relevant_files = [file for file in files if substring in file]
    return relevant_files

########################################################################################################################
def get_label_weights(data, biggest_label, visualize=False):
    true_dis = [0] * (biggest_label+1)
    for y in data.y:
        y = int(y)
        true_dis[y] += 1
    if visualize:
        plt.bar(range(len(true_dis)), true_dis)
        plt.title('true label distribution')
        plt.show()
    class_counts = torch.tensor(true_dis, dtype=torch.float32)
    weights = 1.0 / class_counts
    # Normalize weights
    weights = weights / weights.sum()
    return weights