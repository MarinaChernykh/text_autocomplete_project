import torch
from tqdm import tqdm


def evaluate_rouge_transformer(model, tokenized_texts, scorer, params):
    """Генерирует последнюю четверть текста и определяет средние метрики ROUGE-1 и ROUGE-2
    на валидационной/тестовой выборках."""
    
    device = params["device"]
    tokenizer = params["tokenizer"]

    rouge1_scores = []
    rouge2_scores = []

    for text in tqdm(tokenized_texts):
        if len(text) < 4:
            continue
        
        split_idx = int(len(text) * params["split_ratio"])
        input = text[:split_idx]
        target = text[split_idx:]

        input_tensor = torch.tensor(input, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=params["max_new_tokens"],
                do_sample=True,
                temperature=params["temperature"],
                top_k=params["top_k"],
                pad_token_id=tokenizer.eos_token_id
            )
        generated_seq = generated[0].cpu().tolist()
        pred_seq = generated_seq[len(input):]

        target_text = tokenizer.decode(target, skip_special_tokens=True)
        pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)

        if len(target_text.strip()) == 0 or len(pred_text.strip()) == 0:
            continue

        scores = scorer.score(target_text, pred_text)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0

    return avg_rouge1, avg_rouge2


def show_autocomplete_transformer(model, tokenized_texts, n_samples, params):
    """Демонстрирует n примеров автодополнения текста."""
    device = params["device"]
    tokenizer = params["tokenizer"]
    
    for i in range(n_samples):
        text = tokenized_texts[i]
        split_idx = int(len(text) * params["split_ratio"])
        input = text[:split_idx]
        input_tensor = torch.tensor(input, dtype=torch.long).unsqueeze(0).to(device)
        target = text[split_idx:]
        
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_seq = generated[0].cpu().tolist()

        print("Полный исходный текст      :", tokenizer.decode(text, skip_special_tokens=True))
        print("Полный предсказанный текст :", tokenizer.decode(generated_seq, skip_special_tokens=True))
        print()
        print("Часть для дополнения       :", tokenizer.decode(input, skip_special_tokens=True))
        print("Правильный ответ           :", tokenizer.decode(target, skip_special_tokens=True))
        print("Предсказанный ответ        :", tokenizer.decode(generated_seq[len(input):], skip_special_tokens=True))
        print("===" * 50)
