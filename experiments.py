import torch_geometric.loader
import torch_geometric.transforms
from models.gnns import *
from models.lms import *
from utils import *

import argparse
import gc
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import optuna
import optuna.visualization
import os
from peft import LoraConfig
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import random
import re
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch_sparse import SparseTensor
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import yaml

absolute_path = "/home/yourfirstname.yourlastname/experiment/output_dir_some_name"

def set_random_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataset(dataset_name):
    if is_on_server():
        return PygNodePropPredDataset(name=dataset_name, root="/home/yourfirstname.yourlastname/experiment/dataset")
    else:
        if dataset_name == "ogbn-products":
            return PygNodePropPredDataset(name=dataset_name, root='/Users/yourfirstnameyourlastname/Downloads')
        else:
            return PygNodePropPredDataset(name=dataset_name)

def train_just_mlp(dataset_name='ogbn-arxiv',
                   different_embedding=None,
                   batch_size=16,
                   learning_rate=0.001,
                   max_epoch=30,
                   custom_name=None,
                   save_model_small=False,
                   path_to_pretrained_classifier=None):
    set_random_seed()

    if custom_name == None:
        custom_name = f"bs{batch_size}_lr{learning_rate}_me{max_epoch}"

    dataset = get_dataset(dataset_name=dataset_name)
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    if different_embedding:
        use_different_embedding(data, different_embedding)

    train_data = StandardDataset(data.x[train_mask], data.y[train_mask])
    val_data = StandardDataset(data.x[val_mask], data.y[val_mask])
    test_data = StandardDataset(data.x[test_mask], data.y[test_mask])

    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    text_classifier = TextClassifier(input_dim=data.x[0].shape[0],
                                     hidden_dim=64,
                                     output_dim=num_labels)
    if path_to_pretrained_classifier != None:
        if path_to_pretrained_classifier == "/home/yourfirstname.yourlastname/experiment/classifier_with_standard/classifier_with_standard_bert_products/model_checkpoints/classifier_with_standard_bert_products.pth" or path_to_pretrained_classifier == "/home/yourfirstname.yourlastname/experiment/classifier_with_standard/classifier_with_standard_gpt2_arxiv/model_checkpoints/classifier_with_standard_gpt2_arxiv.pth" or path_to_pretrained_classifier == "/home/yourfirstname.yourlastname/experiment/classifier_with_standard/classifier_with_standard_gpt2_products/model_checkpoints/classifier_with_standard_gpt2_products.pth":
            text_classifier.load_state_dict(torch.load(path_to_pretrained_classifier, map_location="cpu"))
        if not is_on_server() and path_to_pretrained_classifier == "/Users/yourfirstnameyourlastname/Desktop/classifier_with_standard/classifier_with_standard_bert_products/model_checkpoints/classifier_with_standard_bert_products.pth" or path_to_pretrained_classifier == "/Users/yourfirstnameyourlastname/Desktop/classifier_with_standard/classifier_with_standard_gpt2_arxiv/model_checkpoints/classifier_with_standard_gpt2_arxiv.pth" or path_to_pretrained_classifier == "/Users/yourfirstnameyourlastname/Desktop/classifier_with_standard/classifier_with_standard_gpt2_products/model_checkpoints/classifier_with_standard_gpt2_products.pth":
            text_classifier.load_state_dict(torch.load(path_to_pretrained_classifier, map_location="cpu"))
        else:
            text_classifier = torch.load(path_to_pretrained_classifier, map_location="cpu")
    model = LitModel(model=text_classifier,
                     criterion=torch.nn.CrossEntropyLoss(),
                     learning_rate=learning_rate)
    
    logger_dir = "tb_logs"
    model_save_dir = 'model_checkpoints'
    if is_on_server() or True:
        # Tensorboard logging
        logger_dir = os.path.join(absolute_path, logger_dir)
        # saving models
        model_save_dir = os.path.join(absolute_path, model_save_dir)

    create_dir_if_not_exists(logger_dir)
    
    logger = TensorBoardLogger(logger_dir, name=f"mlp_{custom_name}")

    trainer = L.Trainer(max_epochs=max_epoch,
                        log_every_n_steps=1,
                        check_val_every_n_epoch=1,
                        logger=logger,
                        accelerator='cpu',
                        devices=1)
    print("STARTING MLP training", flush=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate the model
    result = trainer.validate(model=model, dataloaders=val_loader)
    val_loss = result[0]['val_loss']
    val_acc = result[0]['val_acc']

    print("STARTING testing", flush=True)
    result = trainer.test(model=model, dataloaders=test_loader)
    test_loss = result[0]['test_loss']
    test_acc = result[0]['test_acc']
    print(f"test_acc: {test_acc}", flush=True)

    if save_model_small:
        create_dir_if_not_exists(model_save_dir)
        dir = model_save_dir
        model_save_name = custom_name if custom_name != "default" else f'lr_{learning_rate}'
        model_save_name += ".pth"
        torch.save(model.model.state_dict(), os.path.join(dir, model_save_name))

    del model
    del train_loader
    del val_loader
    del test_loader

    torch.cuda.empty_cache()
    gc.collect()
    return val_loss

def train_lm_classifier_combi_on_text(model_name='bert-base-uncased',
                                      dataset_name='ogbn-arxiv',
                                      path_to_pretrained_classifier=None,
                                      dropout_rate=0.1,
                                      learning_rate=5e-5,
                                      only_title=False,
                                      noisy=False,
                                      short_train=False,
                                      save_model=False,
                                      save_model_small=False,
                                      max_epoch=1,
                                      custom_name="default"):
    print(f"noisy: {noisy}", flush=True)
    set_random_seed()
    tokenizer = get_tokenizer(model_name)

    dataset = get_dataset(dataset_name=dataset_name)
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    texts = get_text_for_nodes(dataset_name, only_title=only_title, noisy=noisy)['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    if short_train:
        # reduce the size of train data to make hyperparameter tuning faster
        true_at_indices = torch.nonzero(train_mask)
        num_true_to_keep = len(true_at_indices) // 3
        indices_to_keep = random.sample(true_at_indices.tolist(), num_true_to_keep)
        short_train_mask = torch.zeros_like(train_mask, dtype=bool)
        short_train_mask[indices_to_keep] = True
        train_mask = short_train_mask

    train_data = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_data = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_data = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    batch_size = 16 if model_name != "gpt2" else 8

    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    model = LitModel(model=LmWithClassifer(model_name,
                                           num_labels,
                                           tokenizer=tokenizer,
                                           path_to_pretrained_classifier=path_to_pretrained_classifier,
                                           dropout_rate=dropout_rate),
                     criterion=torch.nn.CrossEntropyLoss(),
                     learning_rate=learning_rate)
    
    logger_dir = "tb_logs"
    model_save_dir = 'model_checkpoints'
    if is_on_server() or True:
        # Tensorboard logging
        logger_dir = os.path.join(absolute_path, logger_dir)
        # saving models
        model_save_dir = os.path.join(absolute_path, model_save_dir)

    create_dir_if_not_exists(logger_dir)
    create_dir_if_not_exists(model_save_dir)
    
    logger = TensorBoardLogger(logger_dir, name=f"lm_model_{custom_name}")

    checkpoint_callback = None
    if save_model:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=model_save_dir,
            filename=f'best_lm_{custom_name}',
            save_top_k=1,
            mode='min',
        )
        trainer = L.Trainer(max_epochs=max_epoch,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=1,
                            logger=logger,
                            callbacks=[checkpoint_callback],
                            accelerator='gpu',
                            devices=1)
    else:
        trainer = L.Trainer(max_epochs=max_epoch,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=1,
                            logger=logger,
                            accelerator='gpu',
                            devices=1)
    print("STARTING trainer_fit", flush=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate the model
    result = trainer.validate(model=model, dataloaders=val_loader)
    val_loss = result[0]['val_loss']
    val_acc = result[0]['val_acc']

    if save_model_small:
        dir = model_save_dir
        model_save_name = custom_name if custom_name != "default" else f'lr_{learning_rate}'
        torch.save(model.model.state_dict(), os.path.join(dir, model_save_name))

    print("STARTING testing", flush=True)
    result = trainer.test(model=model, dataloaders=test_loader)
    test_loss = result[0]['test_loss']
    test_acc = result[0]['test_acc']
    print(f"test_acc: {test_acc}", flush=True)

    return val_loss

def testing_lm(model_name='bert-base-uncased',
                dataset_name='ogbn-arxiv',
                dropout_rate=0.1,
                only_title=False,
                noisy=False,
                custom_name="default",
                path_to_model="some_path"):
    set_random_seed()
    tokenizer = get_tokenizer(model_name)

    dataset = get_dataset(dataset_name=dataset_name)
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    texts = get_text_for_nodes(dataset_name, only_title=only_title, noisy=noisy)['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    test_data = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    batch_size = 16 if model_name != "gpt2" else 8

    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    lm=LmWithClassifer(model_name,
                        num_labels,
                        tokenizer=tokenizer,
                        path_to_pretrained_classifier=None,
                        dropout_rate=dropout_rate)
    state_dict = torch.load(path_to_model)
    lm.load_state_dict(state_dict)

    model = LitModel(model=lm,
                     criterion=torch.nn.CrossEntropyLoss(),
                     learning_rate=0)
    
    logger_dir = "tb_logs"
    if is_on_server() or True:
        # Tensorboard logging
        logger_dir = os.path.join(absolute_path, logger_dir)

    create_dir_if_not_exists(logger_dir)
    
    logger = TensorBoardLogger(logger_dir, name=f"custom_name")

    trainer = L.Trainer(max_epochs=1,
                        log_every_n_steps=1,
                        check_val_every_n_epoch=1,
                        logger=logger,
                        accelerator='gpu',
                        devices=1)
    print("STARTING testing", flush=True)
    result = trainer.test(model=model, dataloaders=test_loader)
    test_loss = result[0]['test_loss']
    test_acc = result[0]['test_acc']
    print(f"test_acc: {test_acc}", flush=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': (preds == p.label_ids).astype(np.float32).mean().item(),
    }

def train_hf_lm_classifier(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_on_server():
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="/home/yourfirstname.yourlastname/experiment/dataset")
    else:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    train_dataset = TokenizedDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_dataset = TokenizedDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_dataset = TokenizedDataset(texts[test_mask], data.y[test_mask], tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    result = trainer.evaluate()
    return result

########################################################################################################################
def use_different_embedding(data, path_to_embedding: str):
    if path_to_embedding != "labels":
        loaded_embeddings = np.load(path_to_embedding)
        data.x = torch.tensor(loaded_embeddings, dtype=torch.float)
    else:
        data.x = data.y

def pyg_to_dgl(pyg_data):
    # Create a DGLGraph object from the edge_index
    edge_index = pyg_data.edge_index
    g = dgl.graph((edge_index[0], edge_index[1]))

    # Add node features if available
    if pyg_data.x is not None:
        print("use my x", flush=True)
        g.ndata['feat'] = pyg_data.x

    # Add edge features if available
    if pyg_data.edge_attr is not None:
        g.edata['feat'] = pyg_data.edge_attr

    # # Add labels if available
    # if pyg_data.y is not None:
    #     print(f"use y: {pyg_data.y.shape}")
    #     g.ndata['label'] = pyg_data.y

    return g

# needed for revgat
def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph

def train_graph_model(dataset_name='ogbn-arxiv',
                      hidden_channels=8,
                      num_layers=2,
                      num_heads=8,
                      dropout=0.6,
                      edge_dropout=0.4,
                      input_dropout=0.35,
                      input_norm=False,
                      different_embedding=None,
                      learning_rate=0.005,
                      weight_decay=0.0005,
                      trial_number=9999999,
                      batch_size=1024,
                      test=True,
                      label_smoothing=0.02,
                      number_of_sampled_neighbors=463,
                      only_labels=False,
                      custom_logger_name=None,
                      max_epoch=100,
                      testing=False,
                      use_revgat=False):
    set_random_seed(777)

    dataset = get_dataset(dataset_name=dataset_name)
    data = dataset[0]
    num_nodes = data.num_nodes
    num_labels = len(np.unique(data.y))

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if different_embedding and only_labels:
        raise ValueError("It does not make sense to use a different embedding and only_labels simultaneously")
    elif different_embedding:
        use_different_embedding(data, different_embedding)
    elif only_labels:
        print("only labels!", flush=True)
        use_different_embedding(data, "labels")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # BELOW
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if use_revgat:
        print("using REVGAT!!", flush=True)

        logger_dir = "tb_logs"
        if is_on_server() or True:
            logger_dir = os.path.join(absolute_path, logger_dir) # Tensorboard logging
        if custom_logger_name == None:
            custom_logger_name = f"graph_model_nl{num_layers}hc{hidden_channels}lr={learning_rate}nh={num_heads}bs={batch_size}wd={weight_decay}do={dropout}in={input_norm}"

        g = pyg_to_dgl(data)
        g = preprocess(g)
        model = RevGAT(in_feats=data.x.size(1),
                       n_classes=num_labels,
                       n_hidden=hidden_channels,
                       n_layers=num_layers,
                       n_heads=num_heads,
                       dropout=dropout,
                       input_drop=input_dropout,
                       edge_drop=edge_dropout)
        
        tb_writer = SummaryWriter(os.path.join(logger_dir, custom_logger_name))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        val_frequency = 1

        device = torch.device('cpu')
        print(f"device: {device}")
        model.to(device)

        import time
        start_time = time.time()

        # for epoch in range(1): # just for testing
        for epoch in range(max_epoch):

            epoch_start_time = time.time()

            print(f"Epoch {epoch} / {max_epoch - 1}", flush=True)
            tb_writer.add_scalar("epoch", epoch, epoch)
            # train
            optimizer.zero_grad()
            # print("getting feat", flush=True)
            feat = g.ndata["feat"]
            # print("computing output", flush=True)
            output = model(g.to(device), feat.to(device))
            # print(f"output: {output.shape}", flush=True)
            # print(f"output: {output}", flush=True)
            # print("computing loss", flush=True)
            # print(f"output: {output[train_mask].shape}", flush=True)
            # print(f"y: {data.y[train_mask].squeeze(1).shape}", flush=True)
            loss = criterion(output[train_mask], data.y[train_mask].squeeze(1).to(device))
            # print("loss: {loss}", flush=True)
            tb_writer.add_scalar("training loss", loss, epoch)
            # print("loss backward", flush=True)
            loss.backward()
            # print("optimizer step", flush=True)
            optimizer.step()
            if epoch % val_frequency == 0:
                print("Validation")
                with torch.no_grad():
                    output = model(g.to(device), feat.to(device))
                    val_truth = data.y[val_mask].squeeze(1).to(device)
                    val_loss = criterion(output[val_mask], val_truth)
                    tb_writer.add_scalar("validation loss", val_loss, epoch)
                    val_correct = output[val_mask].argmax(dim=1) == val_truth
                    # print(f"val_correct: {val_correct.shape}", flush=True)
                    # print(f"val_correct: {val_correct}", flush=True)
                    # print(f"int(val_mask.sum()): {int(val_mask.sum())}", flush=True)
                    val_acc = int(val_correct.sum()) / int(val_mask.sum())
                    print(f"validation accuracy: {val_acc}", flush=True)
                    tb_writer.add_scalar("validation accuracy", val_acc, epoch)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            elapsed_time = epoch_end_time - start_time
            estimated_total_time = elapsed_time / (epoch + 1) * max_epoch
            remaining_time = estimated_total_time - elapsed_time
            print(f"Epoch {epoch + 1}/{max_epoch} completed in {epoch_duration:.2f} seconds.", flush=True)
            print(f"Estimated remaining time: {remaining_time / 60:.2f} minutes.\n", flush=True)

        # one final round of testing 
        with torch.no_grad():
            feat = g.ndata["feat"]
            output = model(g.to(device), feat.to(device))
            test_truth = data.y[test_mask].squeeze(1).to(device)
            test_correct = output[test_mask].argmax(dim=1) == test_truth
            test_acc = int(test_correct.sum()) / int(test_mask.sum())
            print(f"testing accuracy: {test_acc}", flush=True)
            tb_writer.add_scalar("test accuracy", val_acc, max_epoch)

        return val_loss 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ABOVE
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Convert the edge index to a sparse adjacency matrix
    if not hasattr(data, 'adj_t'):
        data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])

    # train_loader = torch_geometric.data.DataLoader([data], batch_size=1, shuffle=True)
    # val_loader = torch_geometric.data.DataLoader([data], batch_size=1, shuffle=False)
    # test_loader = torch_geometric.data.DataLoader([data], batch_size=1, shuffle=False)
    # Use NeighborLoader to sample neighbors during training
    num_neighbors = [number_of_sampled_neighbors for _ in range(num_layers)]
    train_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask)
    val_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask)
    test_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.test_mask)

    if use_revgat:
        print("using REVGAT!!", flush=True)
        model = RevGAT(in_feats=data.x.size(1),
                       n_classes=num_labels,
                       n_hidden=hidden_channels,
                       n_layers=num_layers,
                       n_heads=num_heads,
                       dropout=dropout,
                       input_drop=input_dropout,
                       edge_dropout=edge_dropout)
    else:
        model = GATNet2(in_channels=data.x.size(1),
                        hidden_channels=hidden_channels,
                        out_channels=num_labels,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout=dropout,
                        edge_dropout=edge_dropout,
                        input_dropout=input_dropout,
                        input_norm=input_norm)

    logger_dir = "tb_logs"
    accelerator = 'cpu'
    if is_on_server() or True:
        logger_dir = os.path.join(absolute_path, logger_dir) # Tensorboard logging
    if is_on_server():
        accelerator = 'gpu'

    if custom_logger_name == None and only_labels:
        # no batch size different than 1
        custom_logger_name = f"graph_model_nl{num_layers}hc{hidden_channels}lr={learning_rate}nh={num_heads}wd={weight_decay}do={dropout}in={input_norm}"
    elif custom_logger_name == None:
        custom_logger_name = f"graph_model_nl{num_layers}hc{hidden_channels}lr={learning_rate}nh={num_heads}bs={batch_size}wd={weight_decay}do={dropout}in={input_norm}"

    max_epoch = 1 if num_layers == 1 else max_epoch

    if not only_labels: # regular training
        print("not only labels")
        if True or dataset_name == "ogbn-products": # do it this way because of memory footprint
            def print_gpu_memory():
                # Total memory in the GPU
                total_memory = torch.cuda.get_device_properties(0).total_memory
                
                # Allocated memory
                allocated_memory = torch.cuda.memory_allocated(0)
                
                # Cached memory
                cached_memory = torch.cuda.memory_reserved(0)
                
                print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
                print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
                print(f"Cached memory: {cached_memory / (1024 ** 3):.2f} GB")

            tb_writer = SummaryWriter(os.path.join(logger_dir, custom_logger_name))

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            val_frequency = 1

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"device: {device}")
            model.to(device)
            
            for epoch in range(max_epoch):
                print(f"Epoch {epoch} / {max_epoch - 1}")
                tb_writer.add_scalar("epoch", epoch, epoch * len(val_loader))
                # training
                train_batch_id = 0
                for batch in tqdm(train_loader):
                    # print(f"batch: {batch}")
                    train_batch_id += 1
                    # batch = batch.to(device) # batch is too big
                    optimizer.zero_grad()
                    train_mask = batch.train_mask
                    if batch.x.shape[0] == num_nodes: # only move necessary nodes sometimes loader is weird
                        x = batch.x[batch.n_id]
                    else:
                        x = batch.x
                    output = model(x.to(device), batch.edge_index.to(device))
                    loss = criterion(output[train_mask], batch.y[train_mask].squeeze(1).to(device))
                    tb_writer.add_scalar("training loss", loss, epoch * len(train_loader) + train_batch_id)
                    loss.backward()
                    optimizer.step()
                # validation
                if epoch % val_frequency == 0:
                    print("Validation")
                    total_val_loss = 0
                    total_correct = 0
                    val_batch_id = 0
                    with torch.no_grad():
                        for val_batch in tqdm(val_loader):
                            # print(f"val_batch: {val_batch}")
                            val_batch_id += 1
                            # val_batch = val_batch.to(device) # too big
                            val_mask = val_batch.val_mask
                            if val_batch.x.shape[0] == num_nodes: # only move necessary nodes
                                val_x = val_batch.x[val_batch.n_id]
                                # print("used n_id")
                            else: 
                                val_x = val_batch.x
                                # print("used all x")
                            prediction = model(val_x.to(device), val_batch.edge_index.to(device))[val_mask]
                            truth = val_batch.y[val_mask].squeeze(1).to(device)
                            # compute part of total loss
                            val_loss = criterion(prediction, truth)
                            tb_writer.add_scalar("validation loss", val_loss, epoch * len(val_loader) + val_batch_id)
                            total_val_loss += val_loss.item()
                            # compute part of accuracy
                            correct = prediction[:val_batch.batch_size].argmax(dim=1) == truth[:val_batch.batch_size]
                            total_correct += int(correct.sum())
                            # quit()
                        print(f"total correct: {total_correct}")
                        print(f"data.val_mask.sum(): {data.val_mask.sum()}")
                        accuracy = total_correct / int(data.val_mask.sum())
                        print(f"accuracy: {accuracy}")
                        tb_writer.add_scalar("validation accuracy", accuracy, epoch)

            # testing
            if testing:
                print("Testing")
                total_correct_test = 0
                total_correct_test_known = 0
                with torch.no_grad():
                    for test_batch in tqdm(test_loader):
                        test_mask = test_batch.test_mask
                        if test_batch.x.shape[0] == num_nodes: # only move necessary nodes
                            test_x = test_batch.x[test_batch.n_id]
                        else: 
                            test_x = test_batch.x
                        prediction_test = model(test_x.to(device), test_batch.edge_index.to(device))[test_mask]
                        truth_test = test_batch.y[test_mask].squeeze(1).to(device)
                        # compute part of accuracy
                        correct_test = prediction_test[:test_batch.batch_size].argmax(dim=1) == truth_test[:test_batch.batch_size]
                        total_correct_test += int(correct_test.sum())
                        if dataset_name == "ogbn-products": # ignore test cases that were not in training data
                            truth_test_only_known = truth_test.clone()
                            truth_test_only_known[truth_test_only_known >= 42] = -1
                            correct_test_known = prediction_test[:test_batch.batch_size].argmax(dim=1) == truth_test_only_known[:test_batch.batch_size]
                            total_correct_test_known += int(correct_test_known.sum())
                    accuracy_test = total_correct_test / int(data.test_mask.sum())
                    print(f"test accuracy: {accuracy_test}")
                    tb_writer.add_scalar("test accuracy", accuracy_test, 0)
                    if dataset_name == "ogbn-products": # ignore test cases that were not in training data
                        number_of_unknown_examples_in_test_set = 30993 # I precomputed this value
                        accuracy_test_known = total_correct_test_known / (int(data.test_mask.sum()) - number_of_unknown_examples_in_test_set)
                        print(f"test accuracy of known: {accuracy_test_known}")
                        tb_writer.add_scalar("test accuracy of known", accuracy_test_known, 0)

            tb_writer.close()
            # Clean up at the end of the function
            del model
            del train_loader
            del val_loader
            del criterion
            del optimizer
            
            torch.cuda.empty_cache()
            gc.collect()
            return total_val_loss  
        # elif dataset_name == "ogbn-arxiv": # memory okay with lightning
        #     print("loading lit model", flush=True)
        #     litmodel = LitGAT(model,
        #                       torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        #                       learning_rate)
        #     logger = TensorBoardLogger(logger_dir, name=custom_logger_name)
        #     trainer = L.Trainer(max_epochs=max_epoch,
        #                         log_every_n_steps=1,
        #                         check_val_every_n_epoch=1,
        #                         logger=logger,
        #                         accelerator='gpu',
        #                         devices=1)
        #     print("STARTING trainer_fit", flush=True)
        #     trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)
        #     # for hyperparameter tuning
        #     result = trainer.validate(model=litmodel, dataloaders=val_loader)
        #     val_loss = result[0]['val_loss']
        #     print(f'val accuracy={result[0]["val_acc"]}\n', flush=True)
        #     return val_loss
        #     print("STARTING testing", flush=True)
        #     result = trainer.test(model=litmodel, dataloaders=test_loader)
        #     test_acc = result[0]['test_acc']
        #     print(f"test_acc: {test_acc}", flush=True)

    else: # only look at neighboring labels
        print("batch size has to be 1", flush=True)
        train_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1, input_nodes=data.train_mask)
        val_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1, input_nodes=data.val_mask)
        test_loader = torch_geometric.loader.NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1, input_nodes=data.test_mask)

        litmodel = LitGAT_only_labels(model,
                                      torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing),
                                      learning_rate, weight_decay)
        print("using only labels", flush=True)
        logger = TensorBoardLogger(logger_dir, name=custom_logger_name)

        # Setup ModelCheckpoint callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=50,
            mode='min',
        )

        trainer = L.Trainer(max_epochs=1,
                            accelerator=accelerator,
                            devices=1,
                            log_every_n_steps=2,
                            check_val_every_n_epoch=2,
                            logger=logger,
                            callbacks=[early_stopping_callback])
        if custom_logger_name == None:
            print(f'Begin with training trial {trial_number}!\n num_layers: {num_layers}\n hidden_channels={hidden_channels}\n learning_rate={learning_rate}\n num_heads={num_heads}', flush=True)
        trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print("STARTING validation", flush=True)
        result = trainer.validate(model=litmodel, dataloaders=val_loader)
        val_loss = result[0]['val_loss']
        print(f'val accuracy={result[0]["val_acc"]}\n', flush=True)

        print("STARTING testing", flush=True)
        result = trainer.test(model=litmodel, dataloaders=test_loader)
        test_acc = result[0]['test_acc']
        print(f"test_acc: {test_acc}", flush=True)

        return val_loss

########################################################################################################################
def suggest_hyperparameter(trial, name, config):
    param_config = config['hyperparameters'][name]
    if param_config['type'] == 'fixed':
        return param_config['value']
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(name, param_config['choices'])
    elif param_config['type'] == 'float':
        return trial.suggest_float(name, float(param_config['low']), float(param_config['high']), log=param_config.get('log', False))
    elif param_config['type'] == 'loguniform':
        return trial.suggest_loguniform(name, float(param_config['low']), float(param_config['high']))
    elif param_config['type'] == 'int':
        return trial.suggest_int(name, param_config['low'], param_config['high'])

#MLP
def objective0(trial, config):
    dataset_name = config.get("dataset_name", "ogbn-arxiv")
    different_embedding = suggest_hyperparameter(trial, 'different_embedding', config)
    batch_size = suggest_hyperparameter(trial, 'batch_size', config)
    learning_rate = suggest_hyperparameter(trial, 'learning_rate', config)
    max_epoch = suggest_hyperparameter(trial, 'max_epoch', config)

    val_loss = train_just_mlp(dataset_name=dataset_name,
                              different_embedding=different_embedding,
                              batch_size=batch_size,
                              learning_rate=learning_rate,
                              max_epoch=max_epoch)
    trial.report(val_loss, step=0)
    return val_loss

# LM
def objective1(trial, config):
    model_name = config.get('model_name', 'bert-base-uncased')
    dataset_name  = config.get('dataset_name', 'ogbn-arxiv')
    only_title = config.get('only_title', False)
    path_to_pretrained_classifier = suggest_hyperparameter(trial, 'path_to_pretrained_classifier', config)
    dropout_rate = suggest_hyperparameter(trial, 'dropout_rate', config)
    learning_rate = suggest_hyperparameter(trial, 'learning_rate', config)
    max_epochs = suggest_hyperparameter(trial, 'max_epochs', config)

    # Call train_lm_classifier_combi_on_text with the trial's hyperparameters
    val_loss = train_lm_classifier_combi_on_text(model_name=model_name,
                                                 dataset_name=dataset_name,
                                                 path_to_pretrained_classifier=path_to_pretrained_classifier,
                                                 dropout_rate=dropout_rate,
                                                 learning_rate=learning_rate,
                                                 only_title=only_title,
                                                 short_train=False,
                                                 save_model_small=True,
                                                 max_epoch=max_epochs,
                                                 custom_name=f"lr_{learning_rate}")
    trial.report(val_loss, step=0)
    return val_loss

# Graph
def objective2(trial, config):
    different_embedding = config.get("different_embedding", True)
    only_labels = config.get("only_labels", False)
    dataset_name  = config.get('dataset_name', 'ogbn-arxiv')
    # Define the hyperparameters to be tuned
    num_layers = suggest_hyperparameter(trial, 'num_layers', config)
    hidden_channels = suggest_hyperparameter(trial, 'hidden_channels', config)
    learning_rate = suggest_hyperparameter(trial, 'learning_rate', config)
    weight_decay = suggest_hyperparameter(trial, 'weight_decay', config)
    num_heads = suggest_hyperparameter(trial, 'num_heads', config)
    batch_size = suggest_hyperparameter(trial, 'batch_size', config)
    dropout = suggest_hyperparameter(trial, 'dropout', config)
    edge_dropout = suggest_hyperparameter(trial, 'edge_dropout', config)
    input_dropout = suggest_hyperparameter(trial, 'input_dropout', config)
    label_smoothing = suggest_hyperparameter(trial, 'label_smoothing', config)
    input_norm = suggest_hyperparameter(trial, 'input_norm', config)

    if not different_embedding:
        different_embedding = None
    elif is_on_server() and not only_labels:
        different_embedding = '/home/yourfirstname.yourlastname/experiment/saved_embeddings_dir/best_model_lr_1.2452967354253948e-05.npy'
    elif not only_labels:
        different_embedding = 'saved_embeddings/better_bert.npy'
    else:
        different_embedding = None

    number_of_sampled_neighbors = 463 if dataset_name=="ogbn-arxiv" else 26 # 10000
    # Call train_graph_model with the trial's hyperparameters
    val_loss = train_graph_model(dataset_name=dataset_name,
                                 hidden_channels=hidden_channels,
                                 num_layers=num_layers,
                                 num_heads=num_heads,
                                 dropout=dropout,
                                 edge_dropout=edge_dropout,
                                 input_dropout=input_dropout,
                                 input_norm=input_norm,
                                 different_embedding=different_embedding,
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 trial_number=trial.number,
                                 batch_size=batch_size,
                                 test=True,
                                 label_smoothing=label_smoothing,
                                 number_of_sampled_neighbors=number_of_sampled_neighbors,
                                 only_labels=only_labels,
                                 max_epoch=30,
                                 use_revgat=config.get("use_revgat", False))
    trial.report(val_loss, step=0)
    return val_loss

# PEFT
def objective3(trial, config):
    # lora config parameters
    rank = suggest_hyperparameter(trial, 'rank', config) # e.g. 8
    lora_alpha = suggest_hyperparameter(trial, 'lora_alpha', config) # e.g. 32
    lora_dropout = suggest_hyperparameter(trial, 'lora_dropout', config) # e.g. 0.1
    # other parameters
    model_name = config.get('model_name', 'bert-base-uncased')
    learning_rate = suggest_hyperparameter(trial, 'learning_rate', config)
    weight_decay = suggest_hyperparameter(trial, 'weight_decay', config)
    batch_size = suggest_hyperparameter(trial, 'batch_size', config)
    num_train_epochs = suggest_hyperparameter(trial, 'num_train_epochs', config)

    # Call train_lm_classifier_combi_on_text with the trial's hyperparameters
    val_loss = lora_experiment(rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               model_name=model_name,
                               learning_rate=learning_rate,
                               weight_decay=weight_decay,
                               batch_size=batch_size,
                               num_train_epochs=num_train_epochs,
                               save_model=True,
                               custom_name=f"{model_name}_r{rank}la{lora_alpha}ld{lora_dropout}lr{learning_rate}wd{weight_decay}bs{batch_size}nte{num_train_epochs}")
    trial.report(val_loss, step=0)
    return val_loss

# def hyperparameter_tuning(objective, study_name, num_trials=10):
def hyperparameter_tuning(objective, config):
    study_name=config.get('study_name', 'default_study')
    num_trials=config.get('num_trials', 10)
    database_file_name = f"{study_name}.db"
    if is_on_server():
        database_file_name = os.path.join(absolute_path, database_file_name)
    print("creating study", flush=True)
    study = optuna.create_study(study_name=study_name,
                                direction='minimize',
                                storage=f'sqlite:///{database_file_name}',
                                load_if_exists=True)
    def objective_with_config(trial):
        return objective(trial, config)
    
    print("starting study", flush=True)
    study.optimize(objective_with_config, n_trials=num_trials)

    print("Best trial:", flush=True)
    trial = study.best_trial
    print(f"  Value: {trial.value}", flush=True)
    print("  Params: ", flush=True)
    for key, value in trial.params.items():
        print(f"    {key}: {value}", flush=True)

    database_file_dir = 'optuna_plots'
    if is_on_server():
        database_file_dir = os.path.join(absolute_path, database_file_dir)
    create_dir_if_not_exists(database_file_dir)

    # Plot optimization history
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(os.path.join(database_file_dir, 'optimization_history.pdf'))

    # Plot hyperparameter importance
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(os.path.join(database_file_dir, 'optimization_importance.pdf'))

########################################################################################################################
# Distillation
def distill_experiment(model_name="bert-base-uncased",
                       dataset_name="ogbn-arxiv",
                       only_title=False,
                       noisy=False,
                       path_to_teacher_embedding='saved_embeddings/better_bert.npy',
                       min_layers=4, max_layers=9):
    # get training data
    print(f"noisy: {noisy}", flush=True)
    set_random_seed()
    tokenizer = get_tokenizer(name=model_name)

    dataset = get_dataset(dataset_name=dataset_name)
    data = dataset[0]

    teacher_embedding = torch.tensor(np.load(path_to_teacher_embedding), dtype=torch.float)
    data.y = teacher_embedding

    num_labels = 47 # temporary len(np.unique(data.y))

    texts = get_text_for_nodes(dataset_name, only_title, noisy)['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    train_data = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_data = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_data = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    batch_size = 16 if model_name != "gpt2" else 8

    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)

    # test different numbers of layers
    for num_layers in range(min_layers, max_layers+1):
        set_random_seed()
        student_model = create_student_model(num_layers, model_name, tokenizer) # MAKE SURE YOU CAN DO THIS OFFLINE
        no_token_type_ids =  True if model_name == "distilbert-base-uncased" else False
        model = LitDistillationModel(model=student_model,
                     criterion=torch.nn.MSELoss(),
                     learning_rate=0.00005,
                     no_token_type_ids=no_token_type_ids)
    
        logger_dir = "tb_logs"
        model_save_dir = 'model_checkpoints_test'
        if is_on_server() or True:
            # Tensorboard logging
            logger_dir = os.path.join(absolute_path, logger_dir)
            # saving models
            model_save_dir = os.path.join(absolute_path, model_save_dir)

        create_dir_if_not_exists(logger_dir)
        create_dir_if_not_exists(model_save_dir)
        
        logger = TensorBoardLogger(logger_dir, name=f"lm_distill_{num_layers}")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='model_checkpoints',
            filename=f'best_lm_distill_{num_layers}',
            save_top_k=1,
            mode='min',
        )
        trainer = L.Trainer(max_epochs=3,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=1,
                            logger=logger,
                            callbacks=[checkpoint_callback],
                            accelerator='gpu',
                            devices=1)

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        torch.save(model.model.state_dict(), os.path.join(model_save_dir, f'lm_distill_{num_layers}'))

        del model
        del student_model
        torch.cuda.empty_cache()
        gc.collect()

def save_embedding_for_all_distilled(folder_with_models, lm_name="bert-base-uncased", dataset_name="ogbn-arxiv", only_title=False, noisy=False):
    print(f"noisy: {noisy}", flush=True)
    if is_on_server():
        device = "cuda"
    else:
        device = "mps"
    
    model_names = get_file_names_containing_substring(directory=folder_with_models, substring='distill')

    for model_name in model_names:

        tokenizer = get_tokenizer(lm_name)

        match = re.search(r'lm_distill_(\d+)$', model_name)
        if match:
            num_layers = int(match.group(1))
        else:
            print("No number found at the end of the file name.", flush=True)
            continue
        lm = create_student_model(num_layers, lm_name, tokenizer)

        state_dict = torch.load(os.path.join(folder_with_models, model_name))
        lm.load_state_dict(state_dict)

        texts = get_text_for_nodes(dataset_name, only_title, noisy)['text'].to_list()
        if is_on_server() or True:
            file_name = os.path.join(absolute_path, f'distilled_embedding_{num_layers}_layers.npy')
        else:
            file_name = f"saved_embeddings/distilled_embedding_{num_layers}_layers.npy"
        save_text_in_encoded_form(device, lm, tokenizer, texts, file_name)
        del lm
        torch.cuda.empty_cache()
        gc.collect()

def get_distill_baseline(config):
    path_to_embeddings = config.get('path_to_embeddings', None)
    if path_to_embeddings != None:
        embedding_names = get_file_names_containing_substring(path_to_embeddings, 'distilled_embedding')
    else:
        embedding_names = []
    path_to_original_embedding = config.get('path_to_original_embedding', None)
    if path_to_original_embedding != None:
        paths = [path_to_original_embedding]
        names = ['original']
    else:
        paths = []
        names = []
    for embedding_name in embedding_names:
        match = re.search(r'(\d+)_layers', embedding_name)
        if match:
            num_layers = int(match.group(1))
        else:
            print("No number found at the end of the file name.", flush=True)
            continue
        paths.append(os.path.join(path_to_embeddings, embedding_name))
        names.append(f'distilled_{num_layers}')
    for path, name in zip(paths, names):
        train_graph_model(dataset_name=config.get('dataset_name', 'ogbn-arxiv'),
                          hidden_channels=config['graph_model']['hidden_channels'],
                          num_layers=config['graph_model']['num_layers'],
                          num_heads=config['graph_model']['num_heads'],
                          dropout=config['graph_model']['dropout'],
                          edge_dropout=config['graph_model']['edge_dropout'],
                          input_dropout=config['graph_model']['input_dropout'],
                          input_norm=config['graph_model']['input_norm'],
                          different_embedding=path,
                          learning_rate=config['graph_model']['learning_rate'],
                          weight_decay=config['graph_model']['weight_decay'],
                          trial_number=config['graph_model']['trial_number'],
                          batch_size=config['graph_model']['batch_size'],
                          test=config['graph_model']['test'],
                          label_smoothing=config['graph_model']['label_smoothing'],
                          number_of_sampled_neighbors=config['graph_model']['number_of_sampled_neighbors'],
                          custom_logger_name=f'distill_{name}',
                          testing=config.get('testing', False),
                          max_epoch=config.get('max_epoch', 100))

def get_mlp_distill_baseline(config):
    path_to_embeddings = config.get('path_to_embeddings', None)
    if path_to_embeddings != None:
        embedding_names = get_file_names_containing_substring(path_to_embeddings, 'distilled_embedding')
    else:
        embedding_names = []
    path_to_original_embedding = config.get('path_to_original_embedding', None)
    if path_to_original_embedding != None:
        paths = [path_to_original_embedding]
        names = ['original']
    else:
        paths = []
        names = []
    for embedding_name in embedding_names:
        match = re.search(r'(\d+)_layers', embedding_name)
        if match:
            num_layers = int(match.group(1))
        else:
            print("No number found at the end of the file name.", flush=True)
            continue
        paths.append(os.path.join(path_to_embeddings, embedding_name))
        names.append(f'distilled_{num_layers}')
    for path, name in zip(paths, names):
        train_just_mlp(dataset_name=config['dataset_name'],
                       different_embedding=path,
                       batch_size=config["batch_size"],
                       learning_rate=config["learning_rate"],
                       max_epoch=config["max_epoch"],
                       custom_name=f'distill_{name}',
                       save_model_small=config["save_model_small"],
                       path_to_pretrained_classifier=config["path_to_pretrained_classifier"])

#try just a MLP (should not work)
def MLP_distillation(config):
    set_random_seed()
    tokenizer_name = config.get('tokenizer_name', 'bert-base-uncased')
    path_to_teacher_embedding = config.get('path_to_teacher_embedding', 'saved_embeddings/better_bert.npy')
    max_epoch = config.get('max_epoch', 1)

    tokenizer = get_tokenizer(name=tokenizer_name)

    if is_on_server():
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="/home/yourfirstname.yourlastname/experiment/dataset")
    else:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    teacher_embedding = torch.tensor(np.load(path_to_teacher_embedding), dtype=torch.float)
    data.y = teacher_embedding

    # num_labels = len(np.unique(data.y))

    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    train_data = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_data = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_data = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    batch_size = 16 if tokenizer_name != "gpt2" else 8

    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)

    mlp_model = MLP([768], data.y.shape[1])
    model = LitDistillationModel(model=mlp_model,
                    criterion=torch.nn.MSELoss(),
                    learning_rate=0.0001,
                    mlp=True)

    logger_dir = "tb_logs"
    model_save_dir = 'model_checkpoints_test'
    if is_on_server() or True:
        # Tensorboard logging
        logger_dir = os.path.join(absolute_path, logger_dir)
        # saving models
        model_save_dir = os.path.join(absolute_path, model_save_dir)

    create_dir_if_not_exists(logger_dir)
    create_dir_if_not_exists(model_save_dir)
    
    custom_name = "i dont know some name"
    logger = TensorBoardLogger(logger_dir, name=custom_name)

    trainer = L.Trainer(max_epochs=max_epoch,
                        log_every_n_steps=1,
                        check_val_every_n_epoch=1,
                        logger=logger,
                        accelerator='gpu',
                        devices=1)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate the model
    result = trainer.validate(model=model, dataloaders=val_loader)
    val_loss = result[0]['val_loss']

########################################################################################################################
def quantization_experiment(config):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    model_name = config.get('model_name', 'bert-base-uncased')
    if model_name == 'bert-base-uncased':
        config = BertConfig
        config.num_labels = num_labels
        model = BertForSequenceClassification.from_pretrained(model_name)
    else:
        raise ValueError("model_name not supported")
    tokenizer = get_tokenizer(model_name)

    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()
    train_mask, val_mask, _ = get_train_val_test_split(dataset=dataset, size=len(data.y))
    train_dataset = TokenizedDataset(texts[train_mask][:250], data.y[train_mask][:250], tokenizer) # do not forget to remove 25
    eval_dataset = TokenizedDataset(texts[val_mask][:500], data.y[val_mask][:500], tokenizer) # do not forget to remove 25

    unique_log_dir = os.path.join("./quantization_logs", time.strftime("%Y-%m-%d_%H-%M-%S"))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./quantization_results",
        logging_dir=unique_log_dir,
        logging_steps=1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval="epoch",
        report_to="tensorboard",
        seed=123
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()
########################################################################################################################
# PEFT
def lora_experiment(# lora config start
                    rank=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    # lora config end
                    model_name='bert-base-uncased',
                    learning_rate=5e-5,
                    weight_decay=0.01,
                    batch_size=16,
                    num_train_epochs=2,
                    output_dir="my_output_dir_peft",
                    save_model=False,
                    custom_name="default_name_final_model"):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
    set_random_seed()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,  # The rank of the LoRA (Low-Rank Adaptation). This controls the dimensionality of the low-rank matrices used in the adaptation.
        lora_alpha=lora_alpha,  # The scaling factor for the low-rank matrices. It typically adjusts the learning rate for the LoRA parameters.
        lora_dropout=lora_dropout,  # The dropout probability applied to the low-rank matrices during training to prevent overfitting.
    )

    if is_on_server():
        output_dir = absolute_path
    else:
        output_dir = output_dir
    logging_dir = os.path.join(output_dir, "tb_logs")
    logging_dir = os.path.join(logging_dir, custom_name)
    create_dir_if_not_exists(logging_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
    )

    tokenizer = get_tokenizer(model_name)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print(f"Number of classes: {model.config.num_labels}")
    model = get_peft_model(model, peft_config)
    print(f"Number of classes: {model.config.num_labels}")

    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    train_dataset = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_dataset = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_dataset = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    class CustomDataCollator:
        def __init__(self, tokenizer):
            self.data_collator = DataCollatorWithPadding(tokenizer)

        def __call__(self, features):
            inputs = [feature[0] for feature in features]
            labels = torch.tensor([feature[1] for feature in features])
            batch = self.data_collator(inputs)
            batch['labels'] = labels
            return batch

    trainer = Trainer(
        model=model.to("cuda"),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(tokenizer=tokenizer)
    )

    trainer.train()
    if save_model:
        trainer.save_model(os.path.join(output_dir, custom_name))
    
    result = trainer.evaluate(val_dataset)
    return result['eval_loss'] # returns evaluation loss

def quantization_experiment(# lora config start
                    rank=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    # lora config end
                    model_name='bert-base-uncased',
                    learning_rate=5e-5,
                    weight_decay=0.01,
                    batch_size=16,
                    num_train_epochs=2,
                    output_dir="my_output_dir_peft",
                    save_model=False,
                    custom_name="default_name_final_model"):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding
    set_random_seed()

    bits_and_bytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = get_tokenizer(model_name)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, quantization_config=bits_and_bytes_config)

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,  # The rank of the LoRA (Low-Rank Adaptation). This controls the dimensionality of the low-rank matrices used in the adaptation.
        lora_alpha=lora_alpha,  # The scaling factor for the low-rank matrices. It typically adjusts the learning rate for the LoRA parameters.
        lora_dropout=lora_dropout,  # The dropout probability applied to the low-rank matrices during training to prevent overfitting.
    )
    model = get_peft_model(model, peft_config)


    if is_on_server():
        output_dir = absolute_path
    else:
        output_dir = output_dir
    logging_dir = os.path.join(output_dir, "tb_logs")
    logging_dir = os.path.join(logging_dir, custom_name)
    create_dir_if_not_exists(logging_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
    )

    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    train_dataset = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer)
    val_dataset = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer)
    test_dataset = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer)

    class CustomDataCollator:
        def __init__(self, tokenizer):
            self.data_collator = DataCollatorWithPadding(tokenizer)

        def __call__(self, features):
            inputs = [feature[0] for feature in features]
            labels = torch.tensor([feature[1] for feature in features])
            batch = self.data_collator(inputs)
            batch['labels'] = labels
            return batch

    trainer = Trainer(
        model=model.to("cuda"),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(tokenizer=tokenizer)
    )

    trainer.train()
    if save_model:
        trainer.save_model(os.path.join(output_dir, custom_name))
    
    result = trainer.evaluate(val_dataset)
    return result['eval_loss'] # returns evaluation loss

########################################################################################################################
# Probing
def get_hidden_embeddings_for_probing(config):
    def probing_encode_texts(texts, model, tokenizer, device):
        encoded_batch = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors='pt').to(device)
        encoded_batch = {key: value for key, value in encoded_batch.items() if key != 'token_type_ids'}
        with torch.no_grad():
            outputs = model(**encoded_batch)
        # Extract and compute mean hidden states for each layer
        hidden_states = outputs.hidden_states
        mean_hidden_states_per_layer = [layer_hidden_state.mean(dim=1).cpu().numpy() for layer_hidden_state in hidden_states]
        return mean_hidden_states_per_layer

    def probing_save_text_in_encoded_form(device, model, tokenizer, texts: list, save_file_path: str='./saved_embeddings_for_probing'):
        model.to(device)
        text_data = SingleDataset(texts)
        batch_size = 8
        text_loader = DataLoader(text_data, batch_size=batch_size, shuffle=False)
        model.eval()
        encoded_texts = []
        for text in tqdm(text_loader):
            mean_hidden_states_per_layer = probing_encode_texts(text, model, tokenizer, device)
            # later, zip needs equal sizes --> pad last batch in case it is too small
            if mean_hidden_states_per_layer[0].shape[0] < batch_size:
                mean_hidden_states_per_layer = [np.pad(layer, ((0, batch_size - layer.shape[0]), (0, 0)), 'constant') for layer in mean_hidden_states_per_layer]
            # Stack the mean hidden states per layer for the current batch
            encoded_texts.append(mean_hidden_states_per_layer)
        # Transpose the list of lists to get embeddings per layer
        all_embeddings_per_layer = list(map(np.vstack, zip(*encoded_texts)))
        for i, layer_embeddings in enumerate(all_embeddings_per_layer):
            layer_embeddings = layer_embeddings[:len(texts)] # remove the padding
            file_name = os.path.join(save_file_path, f"layer_{i}_embedding.npy")
            np.save(file_name, layer_embeddings)

    lm_name = config.get('lm_name', 'bert-base-uncased')
    if is_on_server():
        device = 'cuda'
        path_to_model = config.get('path_to_model_server')
        save_file_path = absolute_path
    else: 
        device = 'mps'
        path_to_model = config.get('path_to_model')
        save_file_path = './saved_probing_embeddings'
    create_dir_if_not_exists(save_file_path)
    full_model = LmWithClassifer(lm_name=lm_name, num_labels=40)  # full_model = BertClassifier(num_labels=40)
    full_state_dict = torch.load(path_to_model, map_location=device)
    full_model.load_state_dict(full_state_dict, strict=False)
    lm_state_dict = full_model.lm.state_dict()
    # the config is needed to get all hidden states for each layer; that is why i do not use get_lm()
    config = BertConfig.from_pretrained(lm_name)
    config.output_hidden_states = True
    model = BertModel.from_pretrained(lm_name, config=config).to(device)
    model.load_state_dict(lm_state_dict)
    tokenizer = get_tokenizer(name=lm_name)
    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_list()
    print("starting to save embeddings...", flush=True)
    probing_save_text_in_encoded_form(device=device,
                                      model=model,
                                      tokenizer=tokenizer,
                                      texts=texts,
                                      save_file_path=save_file_path
    )

def probing():
    set_random_seed()
    absolute_path = "/home/yourfirstname.yourlastname/experiment/output_dir_probing"
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))

    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))

    if is_on_server():
        path_to_embedding_dir = "/home/yourfirstname.yourlastname/experiment/output_dir_probing_embeddings"
    else:
        path_to_embedding_dir = "/Users/yourfirstnameyourlastname/Desktop/output_dir_probing_embeddings"
    for layer in range(13):
        embeddings = torch.tensor(np.load(os.path.join(path_to_embedding_dir, f"layer_{layer}_embedding.npy")), dtype=torch.float)
        train_data = StandardDataset(embeddings[train_mask], data.y[train_mask])
        val_data = StandardDataset(embeddings[val_mask], data.y[val_mask])

        batch_size = 16

        train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
        val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

        logger_dir = "tb_logs"
        if is_on_server() or True:
            # Tensorboard logging
            logger_dir = os.path.join(absolute_path, logger_dir)
        create_dir_if_not_exists(logger_dir)

        logger = TensorBoardLogger(logger_dir, name=f"layer_{layer}_probing")
        
        trainer = L.Trainer(max_epochs=20,
                            log_every_n_steps=1,
                            check_val_every_n_epoch=1,
                            logger=logger,
                            accelerator='gpu',
                            devices=1)
        
        lit_model = LitModel(model=TextClassifier(input_dim=768, # bert's hidden size
                                              hidden_dim=64,
                                              output_dim=num_labels),
                     criterion=torch.nn.CrossEntropyLoss(),
                     learning_rate=1e-3)

        trainer.fit(model=lit_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

########################################################################################################################
def other_method():
    absolute_path = 'charlie'
    print(absolute_path)
    print()

def print_available_cuda_devices():
    pid = os.getpid()
    print(f"Process ID: {pid}", flush=True)
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}", flush=True)
        for device_id in range(num_devices):
            device_name = torch.cuda.get_device_name(device_id)
            print(f"Device ID: {device_id}, Device Name: {device_name}", flush=True)
    else:
        print("CUDA is not available.", flush=True)

########################################################################################################################

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_single_gpu():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if torch.cuda.is_available():
        available_gpus = list(range(torch.cuda.device_count()))
        if available_gpus:
            # Set CUDA_VISIBLE_DEVICES to the first available GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print(f"Using GPU: {available_gpus[0]}")
        else:
            raise RuntimeError("No available GPUs found.")
    else:
        raise RuntimeError("CUDA is not available.")

if __name__ == '__main__':
    print_available_cuda_devices()
    parser = argparse.ArgumentParser(description="Select which method to run.")
    parser.add_argument('--method', type=str, required=True,
                        choices=['train_lm', 'testing_lm', 'train_hf_lm', 'train_graph',
                                 'tune_hyperparameters', 'distill_experiment', 'distill_embeddings',
                                 'distill_baseline', 'mlp_distill_baseline', 'mlp_distillation', 'save_embedding',
                                 'train_just_mlp', 'quantization_experiment',
                                 'lora_experiment', 'get_hidden_embeddings_for_probing', 
                                 'probing', 'other_method'],
                        help='Method to run: train_lm, train_graph, tune_hyperparameters, other_method')
    parser.add_argument('--config', type=str, help='Path to the configuration file.')

    args = parser.parse_args()
    if (args.method != 'other_method' and args.method != 'probing') and not args.config: # does the function need a config file
        parser.error(f"--config is required for method '{args.method}'")
    elif args.config:
        config = load_config(args.config)
        if 'absolute_path' in config:
            absolute_path = config['absolute_path']
            if not is_on_server():
                absolute_path = absolute_path.replace("/home/yourfirstname.yourlastname/experiment", "/Users/yourfirstnameyourlastname/Desktop/experiment")
            create_dir_if_not_exists(absolute_path)

    if args.method == 'train_lm':
        train_lm_classifier_combi_on_text(model_name=config.get('model_name', 'bert-base-uncased'),
                                          dataset_name=config.get('dataset_name', 'ogbn-arxiv'),
                                          path_to_pretrained_classifier=config.get('path_to_pretrained_classifier', None),#'./saved/classifier_with_standard_bert.pth'),
                                          dropout_rate=config.get('dropout_rate', 0.1),
                                          learning_rate=config.get('learning_rate', 5e-5),
                                          only_title=config.get('only_title', False),
                                          noisy=config.get('noisy', False),
                                          short_train=config.get('short_train', False),
                                          save_model=config.get('save_model', False),
                                          save_model_small=config.get('save_model_small', False),
                                          max_epoch=config.get('max_epoch', 1),
                                          custom_name=config.get('custom_name', 'default'))
    elif args.method == 'testing_lm':
        testing_lm(model_name=config["model_name"],
                   dataset_name=config["dataset_name"],
                   only_title=config["only_title"],
                   noisy=config["noisy"],
                   custom_name=config["custom_name"],
                   path_to_model=config["path_to_model"])
    elif args.method == 'train_hf_lm':
        train_hf_lm_classifier(model_name=config.get('model_name', 'bert-base-uncased'))
    elif args.method == 'train_graph':
        # just for testing
        # train_graph_model(dataset_name=config.get('dataset_name', 'ogbn-arxiv'))
        # quit()
        train_graph_model(dataset_name=config.get('dataset_name', 'ogbn-arxiv'),
                          hidden_channels=config['graph_model']['hidden_channels'],
                          num_layers=config['graph_model']['num_layers'],
                          num_heads=config['graph_model']['num_heads'],
                          dropout=config['graph_model']['dropout'],
                          edge_dropout=config['graph_model']['edge_dropout'],
                          input_dropout=config['graph_model']['input_dropout'],
                          input_norm=config['graph_model']['input_norm'],
                          different_embedding=config['graph_model']['different_embedding'],
                          learning_rate=config['graph_model']['learning_rate'],
                          weight_decay=config['graph_model']['weight_decay'],
                          trial_number=config['graph_model']['trial_number'],
                          batch_size=config['graph_model']['batch_size'],
                          test=config['graph_model']['test'],
                          label_smoothing=config['graph_model']['label_smoothing'],
                          number_of_sampled_neighbors=config['graph_model']['number_of_sampled_neighbors'],
                          only_labels=config['graph_model']['only_labels'],
                          custom_logger_name=config['graph_model']['custom_logger_name'],
                          testing=config['graph_model']['testing'],
                          max_epoch=config.get('max_epoch', 100))
    # hyperparameter tuning
    elif args.method == 'tune_hyperparameters':
        objective_str = config.get('objective', 'objective1')
        if objective_str == 'objective0':
            hyperparameter_tuning(objective=objective0, config=config)
        if objective_str == 'objective1':
            hyperparameter_tuning(objective=objective1, config=config)
            #hyperparameter_tuning(objective=objective1, study_name=config.get('study_name', 'default_study'))
        elif objective_str == 'objective2':
            hyperparameter_tuning(objective=objective2, config=config)
            #hyperparameter_tuning(objective=objective2, study_name=config.get('study_name', 'default_study'), num_trials=config.get('num_trials', 100))
        elif objective_str == 'objective3':
            if is_on_server(): set_single_gpu()
            hyperparameter_tuning(objective=objective3, config=config)
    # distillation
    elif args.method == 'distill_experiment':
        model_name = config.get('model_name', 'bert-base-uncased')
        dataset_name = config.get('dataset_name', 'ogbn-arxiv')
        only_title = config.get('only_title', False)
        noisy = config.get('noisy', False)
        path_to_teacher_embedding = config.get('path_to_teacher_embedding', 'saved_embeddings/better_bert.npy')
        min_layers = config.get('min_layers', 4)
        max_layers = config.get('max_layers', 9)
        distill_experiment(model_name=model_name, dataset_name=dataset_name, only_title=only_title, noisy=noisy, path_to_teacher_embedding=path_to_teacher_embedding, min_layers=min_layers, max_layers=max_layers)
    elif args.method == 'distill_embeddings':
        if is_on_server:
            file_path = config.get('file_path', '/home/yourfirstname.yourlastname/experiment/does_not_exist')
        else:
            file_path = config.get('file_path', '/home/yourfirstname.yourlastname/experiment/does_not_exist')
        model_name = config.get('model_name', 'bert-base-uncased')
        dataset_name = config.get('dataset_name', 'ogbn-arxiv')
        only_title = config.get('only_title', False)
        noisy = config.get('noisy', False)
        save_embedding_for_all_distilled(folder_with_models=file_path, lm_name=model_name, dataset_name=dataset_name, only_title=only_title, noisy=noisy)
    elif args.method == 'distill_baseline':
        get_distill_baseline(config)
    elif args.method == 'mlp_distill_baseline':
        get_mlp_distill_baseline(config)
    elif args.method == 'mlp_distillation':
        MLP_distillation(config)
    # saving embedding
    elif args.method == 'save_embedding':
        lm_name = config.get('lm_name', 'bert-base-uncased')
        use_standard = config.get('use_standard', False)
        dataset_name = config.get('dataset_name', 'ogbn-arxiv')
        only_title = config.get('only_title', False)
        noisy = config.get('noisy', False)
        print(f"noisy: {noisy}", flush=True)
        use_lightning_checkpoint = config.get('use_lightning_checkpoint', False)
        map_location = 'cuda' if is_on_server() else 'mps'
        tokenizer, model = get_tokenizer_and_lm(name=lm_name)
        if use_standard:
            print(f"cuda is available in if: {torch.cuda.is_available()}", flush=True)
        else:
            if dataset_name == "ogbn-arxiv":
                full_model = LmWithClassifer(lm_name=lm_name, num_labels=40, tokenizer=tokenizer) # old: full_model = BertClassifier(num_labels=40)
            elif dataset_name == "ogbn-products":
                full_model = LmWithClassifer(lm_name=lm_name, num_labels=47, tokenizer=tokenizer)
            if use_lightning_checkpoint:
                checkpoint = torch.load(config.get('path_to_model'), map_location=map_location)
                full_state_dict = checkpoint['state_dict']
            else:
                full_state_dict = torch.load(config.get('path_to_model'))
            full_model.load_state_dict(full_state_dict, strict=False)
            lm_state_dict = full_model.lm.state_dict()
            model.load_state_dict(lm_state_dict)
        texts = get_text_for_nodes(dataset_name, only_title, noisy)['text'].to_list()
        print("starting to save embeddings..", flush=True)
        file_name = config.get('path_to_save_file')
        full_save_file_path = config.get('full_save_file_path_given', False)
        if full_save_file_path:
            create_dir_if_not_exists(full_save_file_path)
            file_name = os.path.join(full_save_file_path, file_name)
        else:
            if is_on_server() or True:
                file_name = os.path.join(absolute_path, file_name)
        if is_on_server():
            print("create tensor before function call", flush=True)
            tensor = torch.randn(3, 3)
            print("moving tensor before function call", flush=True)
            tensor = tensor.to("cuda")
            print("moved tensor before function call", flush=True)

            device = config.get('device', 'cuda')
        else:
            device = "mps"
        save_text_in_encoded_form(device=device,
                                  model=model,
                                  tokenizer=tokenizer,
                                  texts=texts,                 
                                  file_name=file_name
        )
    elif args.method == 'train_just_mlp':
        dataset_name: str = config.get('dataset_name', 'ogbn-arxiv')
        different_embedding = config.get('different_embedding', None)
        batch_size: int = config.get('batch_size', 16)
        learning_rate: float = config.get('learning_rate', 0.001)
        max_epoch: int = config.get('max_epoch', 30)
        custom_name = config.get('custom_name', None)
        save_model_small: bool = config.get('save_model_small', False)
        path_to_pretrained_classifier = config.get('path_to_pretrained_classifier', None)
        train_just_mlp(dataset_name=dataset_name,
                       different_embedding=different_embedding,
                       batch_size=batch_size,
                       learning_rate=learning_rate,
                       max_epoch=max_epoch,
                       custom_name=custom_name,
                       save_model_small=save_model_small,
                       path_to_pretrained_classifier=path_to_pretrained_classifier)
    elif args.method == 'quantization_experiment':
        quantization_experiment(config)
    elif args.method == 'lora_experiment':
        lora_experiment(config)
    elif args.method == 'get_hidden_embeddings_for_probing':
        get_hidden_embeddings_for_probing(config)
    elif args.method == 'probing':
        probing()
    elif args.method == 'other_method':
        other_method()
        print(f'outside: {absolute_path}')
        absolute_path = 'bob'
        other_method()
        print(f'outside: {absolute_path}')


# python experiments.py --method train_lm --config config/train_lm_config.yaml
# python experiments.py --method train_lm --config config/train_lm_products_config.yaml
# python experiments.py --method train_graph --config config/train_graph_config.yaml
# python experiments.py --method tune_hyperparameters --config config/tune_hyperparameters_config.yaml
# python experiments.py --method tune_hyperparameters --config config/tune_hyperparameters_graph_config.yaml
# python experiments.py --method distill_experiment
# python experiments.py --method save_embedding --config config/save_embedding_config.yaml
# python experiments.py --method distill_baseline --config config/distill_baseline_config.yaml
# python experiments.py --method quantization_experiment --config config/quantization_config.yaml

    # trial_num = int(sys.argv[1]) # Read trial number from the command

    # train_lm_classifier_combi_on_text(path_to_pretrained_classifier='./saved/classifier_with_standard_bert.pth',
    #                                   dropout_rate=0.06931703566828175,
    #                                   learning_rate=2.0372794815852712e-05,
    #                                   short_train=False,
    #                                   save_model=True)

                                      
    #train_graph_model(different_embedding="saved_embeddings/better_bert.npy")
    # hyperparameter_tuning(objective1, f'my_study') # f'my_study_{trial_num}')

    # # saving embedding
    # print("lets go")
    # device = "mps"

    # state_dict = torch.load('model_checkpoints/best_model.pth')
    # bert_state_dict = {k: v for k, v in state_dict.items() if 'bert' in k}

    # lm = BertModel.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # texts = get_text_for_nodes('ogbn-arxiv')['text'].to_list()
    # file_name = "saved_embeddings/better_bert_hyper_tuned.npy"
    # save_text_in_encoded_form(device, lm, tokenizer, texts, file_name)


# BEST:
# Trial 3 finished with value: 0.9875100255012512 and parameters: 
# {'path_to_pretrained_classifier': './saved/classifier_with_standard_bert.pth', 
#'dropout_rate': 0.06931703566828175, 'learning_rate': 2.0372794815852712e-05}. Best is trial 3 with value: 0.9875100255012512.


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃      Validate metric      ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │          val_acc          │    0.7136816382408142     │
# │         val_loss          │    0.9259032011032104     │
# └───────────────────────────┴───────────────────────────┘
