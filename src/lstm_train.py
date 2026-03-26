import torch
from tqdm import tqdm


def train_epoch(model, dataloader, params):   
    device = params["device"]
    tokenizer = params["tokenizer"]
    optimizer = params["optimizer"]
    criterion = params["criterion"]
    
    model.train()

    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader):
        x = batch["texts"].to(device)
        y = batch["answers"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clipping"])
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=-1)
        mask = (y != tokenizer.pad_token_id)
        total_correct += ((preds == y) & mask).sum().item()
        total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy
