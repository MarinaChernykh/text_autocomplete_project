import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AutocompleteDataset(Dataset):
    """Класс для создания датасетов. При обращении к элементу датасета по индексу получаем пару для обучения:
        'text' - текст без последнего токена
        'answer' - текст без первого токена
        т.е. 2 последовательности, смещенные друг относительно друга на 1 токен
    """
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_idx = self.texts[idx]
        return {
            "text": torch.tensor(text_idx[:-1], dtype=torch.long),
            "answer": torch.tensor(text_idx[1:], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    """Дополняет элементы батча до единой длины с помощью паддинга."""
    texts = [item["text"] for item in batch]
    answers = [item["answer"] for item in batch]
    lengths = torch.tensor([len(item["text"]) for item in batch])
    
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_token_id)
    padded_answers = pad_sequence(answers, batch_first=True, padding_value=pad_token_id)
    
    return {
        "texts": padded_texts,
        "answers": padded_answers,
        "lengths": lengths
    }
