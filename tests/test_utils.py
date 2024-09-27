import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import BertTokenizer, BertModel

from experiments import *
from utils import *
from models.lms import *

def test_save_text_in_encoded_form():
    device = "mps"
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = ["This is the 1. text",
             "This is the 2. text",
             "This is the 3. text",
             "This is the 4 text",
             "This is the 5. text",
             "This is the 6. text",
             "This is the 7. text",
             "This is the 8. text",
             "This is the 9. text",
             "This is the 10. text",
             "This is the 11. text",
             "This is the 12. text",]
    file_name = "tests/mybadfilename.npy"
    save_text_in_encoded_form(device, model, tokenizer, texts, file_name)
    loaded_embeddings = np.load(file_name)
    assert loaded_embeddings.shape == (12, 768)

def test_tokenized_standard_dataset():
    set_random_seed()
    tokenizer = get_tokenizer()
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    num_labels = len(np.unique(data.y))
    texts = get_text_for_nodes('ogbn-arxiv')['text'].to_numpy()
    train_mask, val_mask, test_mask = get_train_val_test_split(dataset=dataset, size=len(data.y))
    train_data = TokenizedStandardDataset(texts[train_mask], data.y[train_mask], tokenizer, 100)
    val_data = TokenizedStandardDataset(texts[val_mask], data.y[val_mask], tokenizer, 100)
    test_data = TokenizedStandardDataset(texts[test_mask], data.y[test_mask], tokenizer, 100)
    batch_size = 2
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = torch_geometric.loader.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)
    for encoded, y in train_loader:
        assert encoded['input_ids'].shape == (2, 100)
        assert y.shape == (2, 1)
        break

# if __name__ == "__main__":
#     test_tokenized_standard_dataset()

# pytest tests/test_utils.py