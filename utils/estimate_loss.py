import torch
from torch.amp import autocast
from layers.GPTLanguageModel import GPTLanguageModel
from utils.get_batch import get_batch

EVAL_ITERS = 50 # TODO CAN REDUCE TO INCREASE SPEED OF THE PROCESS OR MAKE THE EVAL PROCESS LESS FREQUENT

@torch.no_grad()
def estimate_loss(model : GPTLanguageModel, train_data, val_data, batch_size):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, batch_size, model.blockSize, next(model.parameters()).device)
            with autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
