import torch
from transformers import AutoTokenizer

TRAIN_TEST_SPLIT = 0.9
def get_train_val_tensors(text : str, tokenizer : AutoTokenizer, train_test_split : float = TRAIN_TEST_SPLIT):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_test_split * len(data))
    train_data, val_data = data[:n], data[n:]
    del data
    torch.cuda.empty_cache()

    return train_data, val_data
