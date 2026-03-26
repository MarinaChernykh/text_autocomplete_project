import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AutocompleteLSTM(nn.Module):
    """Класс, описывающий LSTM модель для генерации текста."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, input_ids, lengths=None):
        embedded = self.embedding(input_ids)
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(embedded)
        
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
    
    def generate(self, input_ids, max_new_tokens=30, eos_token_id=None):
        """Генерирует токены до конца строки или до достижения указанного кол-ва."""
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self(generated)
                
            next_token_logits = logits[:, -1, :]  # последний токен
            # базовый вариант
            # next_token = torch.argmax(next_token_logits, dim=-1)
            # второй вариант
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # третий вариант
            # temperature = 0.8
            # probs = torch.softmax(next_token_logits / temperature, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            if eos_token_id is not None:
                if next_token.item() == eos_token_id:
                    break

        return generated
