from utils import *

import math

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR 

import pytorch_lightning as L

from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

from transformers import BertModel, BertTokenizer, BertConfig, DistilBertModel, GPT2Tokenizer, GPT2Model, AutoConfig, AutoModel

class MLP(nn.Module):
    def __init__(self, hidden_dims, output_dim):
        super(MLP, self).__init__()
        # Load pre-trained BERT model
        bert = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = bert.embeddings.word_embeddings
        self.layers = nn.ModuleList()

        embedding_dim = bert.config.hidden_size  # 768
        sequence_length = bert.config.max_position_embeddings  # 512
        input_dim = sequence_length * embedding_dim  # 512 * 768 = 393216
        
        # Input layer
        if hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # Output layer
        if hidden_dims:
            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.view(x.size(0), -1)  # Flatten the embedding output if necessary
        
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x


def get_tokenizer(name='bert-base-uncased'):   
    if name == 'bert-base-uncased' or name == 'distilbert-base-uncased':
        if is_on_server():
            # offline version
            return BertTokenizer.from_pretrained('saved_hugging_face/bert-base-uncased-tokenizer')
        else:
            # online version
            return BertTokenizer.from_pretrained('bert-base-uncased')
    elif name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        return tokenizer

def get_lm(name='bert-base-uncased', config=None):
    if is_on_server():
        # offline version
        if name == 'bert-base-uncased':
            if config == None:
                return BertModel.from_pretrained('saved_hugging_face/bert-base-uncased')
            else:
                return BertModel.from_pretrained('saved_hugging_face/bert-base-uncased', config=config)
        elif name == 'distilbert-base-uncased':
            return DistilBertModel.from_pretrained('saved_hugging_face/distilbert-base-uncased')
    else:
        # online version
        if name == 'bert-base-uncased':
            return BertModel.from_pretrained('bert-base-uncased')
        elif name == 'distilbert-base-uncased':
            return DistilBertModel.from_pretrained('distilbert-base-uncased')
    # TODO offline version of gpt2
    if name == "gpt2":
        return GPT2Model.from_pretrained("gpt2")

def get_tokenizer_and_lm(name='bert-base-uncased', config=None):
    tokenizer = get_tokenizer(name)
    lm = get_lm(name, config)
    if name == "gpt2":
        lm.resize_token_embeddings(len(tokenizer))
        lm.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, lm

def create_student_model(num_layers, model_name='bert-base-uncased', tokenizer=None):
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = num_layers
    elif hasattr(config, 'n_layer'):
        config.n_layer = num_layers
    student_model = AutoModel.from_pretrained(model_name, config=config)
    if model_name == "gpt2":
        student_model.resize_token_embeddings(len(tokenizer))
        student_model.config.pad_token_id = tokenizer.pad_token_id
    return student_model

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First linear layer
            nn.ReLU(),                         # ReLU activation
            nn.Linear(hidden_dim, output_dim)  # Output layer
        )
        
    def forward(self, x):
        return self.network(x)

class BertClassifier(nn.Module):
    def __init__(self, num_labels, path_to_pretrained_classifier=None, dropout_rate=0.1, device=torch.device("cuda")):
        super(BertClassifier, self).__init__()
        # Load a pre-trained BERT model
        self.bert = get_lm()
        # A dropout layer for some regularization
        self.dropout = nn.Dropout(dropout_rate)
        # set up classifier
        if path_to_pretrained_classifier == None:
            self.classifier = TextClassifier(input_dim=self.bert.config.hidden_size, hidden_dim=64, output_dim=num_labels)
        else:
            if is_on_server():
                device = torch.device("cuda")
            else:
                device = torch.device("mps")
            self.classifier = torch.load(path_to_pretrained_classifier, map_location=device)
        # # A tokenizer for Bert
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def forward(self, encoded_input):
        # Feed encoded input to BERT
        outputs = self.bert(**encoded_input)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Pass the output through the classifier to get the predictions
        logits = self.classifier(pooled_output)
        return logits

class LmWithClassifer(nn.Module):
    def __init__(self, lm_name, num_labels, tokenizer=None, path_to_pretrained_classifier=None, dropout_rate=0.1, lora_config=None):
        super(LmWithClassifer, self).__init__()
        self.lm = get_lm(lm_name)
        self.decoder = False
        if lm_name == "gpt2":
            self.decoder = True
            self.lm.resize_token_embeddings(len(tokenizer))
            self.lm.config.pad_token_id = tokenizer.pad_token_id
        if lora_config != None:
            self.lm = get_peft_model(self.lm, lora_config)
        self.no_token_type_ids = False
        if lm_name == 'distilbert-base-uncased':
            self.no_token_type_ids = True
        # A dropout layer for some regularization
        self.dropout = nn.Dropout(dropout_rate)
        # set up classifier
        # old way
        # if path_to_pretrained_classifier == None:
        #     self.classifier = TextClassifier(input_dim=self.lm.config.hidden_size, hidden_dim=64, output_dim=num_labels)
        # else:
        #     if is_on_server():
        #         device = torch.device("cuda")
        #     else:
        #         device = torch.device("mps")
        #     self.classifier = torch.load(path_to_pretrained_classifier, map_location=device)
        self.classifier = TextClassifier(input_dim=self.lm.config.hidden_size, hidden_dim=64, output_dim=num_labels)
        if is_on_server():
            device = torch.device("cuda")
        else:
            device = torch.device("mps")
        if path_to_pretrained_classifier != None:
            self.classifier.load_state_dict(torch.load(path_to_pretrained_classifier, map_location=device))

        
    def forward(self, encoded_input):
        # Feed encoded input to LM
        if self.no_token_type_ids:
            encoded_input = {key: value for key, value in encoded_input.items() if key != 'token_type_ids'}
        outputs = self.lm(**encoded_input)
        if self.decoder: # e.g. gpt2
            hidden_states = outputs[0]
            input_ids = encoded_input['input_ids']
            batch_size, sequence_length = input_ids.shape
            sequence_lengths = torch.eq(input_ids, self.lm.config.pad_token_id).int().argmax(-1) - 1 # where does padding start
            sequence_lengths = sequence_lengths % input_ids.shape[-1] # cannot larger than input
            sequence_lengths = sequence_lengths.to(hidden_states.device) #  move to right device
            last_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            last_hidden_state = self.dropout(last_hidden_state)
            logits = self.classifier(last_hidden_state)
            return logits
        else: # e.g. bert
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            # Apply dropout
            pooled_output = self.dropout(pooled_output)
            # Pass the output through the classifier to get the predictions
            logits = self.classifier(pooled_output)
            return logits

class LitModel(L.LightningModule):
    def __init__(self, model, criterion, learning_rate):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(1))
        # Call update_and_allocate() method needed for PEFT
        # self.model.lm.update_and_allocate()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(1))
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y.squeeze(1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}
        # self.log('val_loss', loss)
        # return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(1))
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y.squeeze(1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Example scheduler
        return optimizer # [optimizer], [scheduler]
    
class LitDistillationModel(L.LightningModule):
    def __init__(self, model, criterion, learning_rate, mlp=False, no_token_type_ids=False):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.mlp = mlp
        self.no_token_type_ids = no_token_type_ids

    def forward(self, x):
        if self.no_token_type_ids:
            encoded_input = {key: value for key, value in x.items() if key != 'token_type_ids'}
            return self.model(**encoded_input)
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if not self.mlp:
            y_hat = self(x).last_hidden_state.mean(dim=1)
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if not self.mlp:
            y_hat = self(x).last_hidden_state.mean(dim=1)
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if not self.mlp:
            y_hat = self(x).last_hidden_state.mean(dim=1)
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Example scheduler
        return optimizer # [optimizer], [scheduler]

########################################################################################################################
# class BertClassifierWithPromptTuning(nn.Module):
#     def __init__(self, device, num_labels, prompt_length=20, path_to_pretrained_classifier=None):
#         super(BertClassifierWithPromptTuning, self).__init__()
#         self.device = device

#         # Load a pre-trained BERT model
#         self.bert = BertModel.from_pretrained('bert-base-uncased')

#         # Number of prompt tokens
#         self.prompt_length = prompt_length
#         # Trainable prompt embeddings
#         self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, self.bert.config.hidden_size))

#         # A dropout layer for some regularization
#         self.dropout = nn.Dropout(0.1)

#         if path_to_pretrained_classifier == None:
#             self.classifier = SimpleMLPClassifier(input_dim=self.bert.config.hidden_size, hidden_dim=64, output_dim=num_labels)
#         else:
#             self.classifier = torch.load(path_to_pretrained_classifier)

#         # A tokenizer for Bert
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
#     def forward(self, input_text):
#         encoded_input = self.tokenizer(input_text, return_tensors='pt', padding="max_length", truncation=True, max_length=512-self.prompt_length).to(self.device)
#         input_ids = encoded_input['input_ids']
#         attention_mask = encoded_input['attention_mask']

#         # Generate prompt embeddings and repeat across batch
#         batch_size = input_ids.size(0)
#         prompt_embeddings = self.prompt_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # same for every batch entry
        
#         # Get input embeddings from BERT
#         inputs_embeds = self.bert.embeddings(input_ids)
#         # Concatenate prompt embeddings with the input embeddings
#         full_embeddings = torch.cat((prompt_embeddings, inputs_embeds), dim=1)

#         # Extend the attention mask for the prompts
#         extended_attention_mask = torch.cat(
#             [torch.ones(batch_size, self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
#             dim=1
#         )

#         # Feed input to BERT
#         outputs = self.bert(inputs_embeds=full_embeddings, attention_mask=extended_attention_mask)
#         pooled_output = outputs.last_hidden_state.mean(dim=1)
#         # Apply dropout
#         pooled_output = self.dropout(pooled_output)
#         # Pass the output through the classifier to get the predictions
#         logits = self.classifier(pooled_output)
#         return logits
    
########################################################################################################################
# def train_with_text(model, optimizer, criterion, x, y):
#     model.train()
#     optimizer.zero_grad()
#     out = model(x)
#     loss = criterion(out, y.squeeze(1))
#     loss.backward()
#     optimizer.step()
#     return loss

# def evaluate_with_text(model, x, y, label_distribution=None):
#     model.eval()
#     with torch.no_grad():
#         out = model(x)
#         pred = out.argmax(dim=1)  # Use the class with highest probability
#         correct = pred == y.squeeze(1)
#         num_correct = int(correct.sum())
#         if label_distribution == None:
#             return num_correct
#         else:
#             for p_t in pred:
#                 p = int(p_t)
#                 label_distribution[p] += 1
#             return num_correct, label_distribution

########################################################################################################################
# below is irrelevant
########################################################################################################################
# class LoRAAttention(nn.Module):
#     def __init__(self, embed_dim, rank, device):
#         super(LoRAAttention, self).__init__()
#         self.rank = rank
#         self.embed_dim = embed_dim

#         # Low-rank matrices A and B
#         self.A = nn.Parameter(torch.Tensor(embed_dim, rank).to(device))
#         self.B = nn.Parameter(torch.Tensor(rank ,embed_dim).to(device))
#         nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

#     def forward(self, W):
#         # LoRA adjustment
#         delta_W = self.A @ self.B
#         return W + delta_W

# def apply_lora_to_bert(bert_model, rank=4, device='cuda'):
#     # First pass: collect all applicable linear layers
#     layers_to_adapt = []
#     for name, module in bert_model.named_modules():
#         if isinstance(module, nn.Linear) and 'attention' in name:
#             layers_to_adapt.append((name, module))

#     # Second pass: apply LoRA adjustments
#     for name, module in layers_to_adapt:
#         lora_att = LoRAAttention(module.weight.shape[1], rank, device)
#         original_weight = module.weight.detach()
        
#         # Define a new forward function
#         def new_forward(x, module=module, lora_att=lora_att):
#             adjusted_weight = lora_att(original_weight)
#             return nn.functional.linear(x, adjusted_weight, module.bias)

#         # Update the forward method
#         setattr(module, 'forward', new_forward)
#         setattr(bert_model, f'{name}_lora', lora_att)

########################################################################################################################
# if __name__ == "__main__":
#     # Example of using normal bert
#     # Instantiate the model
#     model = BertClassifier(num_labels=2)  # For binary classification
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     text = "Example text for classification"

#     # Encode the text using the tokenizer
#     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']

#     # Forward pass
#     with torch.no_grad():
#         logits = model(input_ids, attention_mask)

#     print(logits)
#     ###############################################################################################################################################
#     # Example of using bert with prompt tuning
#     model = BertClassifierWithPromptTuning(num_labels=2, prompt_length=20)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     text = "Example text for classification"

#     # Encode the text using the tokenizer
#     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512-model.prompt_length)
#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']

#     # Forward pass
#     with torch.no_grad():
#         logits = model(input_ids, attention_mask)
#     print(logits)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CrossEntropyLoss Requirement: torch.nn.CrossEntropyLoss expects class indices, not one-hot vectors.
# CrossEntropyLoss Requirement: torch.nn.CrossEntropyLoss expects class indices, not one-hot vectors.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
