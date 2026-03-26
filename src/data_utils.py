import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_dataset(path):
    """Загружает датасет из файла."""
    try:
        df = pd.read_csv(path)
        print(f"Количество записей в датасете: {df.shape[0]}\n")
        print("Первая запись из датасета:")
        print(f"{df['tweet'][0]}\n")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Файл {path.split('/')[-1]} не найден.")
    except Exception as e:
        print(f"При открытии файла {path.split('/')[-1]} произошла ошибка.")


def save_dataset(data, path):
    """Сохраняет тексты в scv файл."""
    try:
        df = pd.DataFrame(data, columns=['tweet'])
        df.to_csv(path)
        print(f"Файл {path.split('/')[-1]} успешно сохранен")
    except Exception as e:
        print(f"При сохранении файла {path.split('/')[-1]} произошла ошибка.")


def clean_text(text):
    """Очищает тексты от упоминаний, хештегов, ссылок, лишних символов."""
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|https\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def show_statistics(texts):
    """Отображает статистическую информацию о текстах датасета."""
    word_counts = [len(text.split()) for text in texts]
    print("\nСтатистика по количеству слов в тексте:")
    print(f"Среднее: {np.mean(word_counts):.2f}")
    print(f"Медиана: {np.median(word_counts):.2f}")
    print(f"Минимальное: {np.min(word_counts):.2f}")
    print(f"Максимальное: {np.max(word_counts):.2f}")
    print(f"5-й перцентиль: {np.percentile(word_counts, 5):.2f}")
    print(f"95-й перцентиль: {np.percentile(word_counts, 95):.2f}")
    
    plt.hist(word_counts, bins=50, edgecolor='black')
    plt.title("Распределение количества слов в текстах")
    plt.xlabel("Количество слов")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()
    
    counter = Counter(" ".join(texts).split())
    print("Топ-20 самых популярных слов в текстах:")
    print(counter.most_common(20))
    print(f"Количество уникальных слов: {len(counter)}")


def clean_short_tweet(texts, min_text_lenght):
    """Удаляет тексты, кол-во слов в которых менее указанного."""
    texts = [text for text in texts if len(text.split()) >= min_text_lenght]
    return texts
  

def split_data(data, val_size, test_size, random_state):
    """Разделяет данные на обучающую, валидационную и тестовую выборки."""
    train, val_test = train_test_split(
        data,
        test_size=(val_size+test_size),
        random_state=random_state
    )
    val, test = train_test_split(
        val_test,
        test_size=(test_size/(val_size+test_size)),
        random_state=random_state
    )
    print(f"Обучающая выборка: {len(train)} записей")
    print(f"Валидационная выборка: {len(val)} записей")
    print(f"Тестовая выборка: {len(test)} записей")
    
    return train, val, test


def tokenize_texts(texts, tokenizer):
    """Токенизирует тексты и добавляет спец. токены начала и окончания текста."""
    
    tokenized_texts = tokenizer(
        texts,
        truncation=True,
        add_special_tokens=True
    )
    return tokenized_texts["input_ids"]


# def prepair_targets(texts, seq_length):
#     X, y = [], []
#     for text in texts:
#         if len(text) > seq_length:
#             for i in range(len(text) - seq_length):
#                 X.append(text[i:seq_length+i])
#                 y.append(text[i+1:seq_length+i+1])
#     print("Пример данных со смещением")
#     print(f"X: {X[0]}")
#     print(f"y: {X[1]}")
#     return X, y


# def split_data(X, y, random_state):
#     X_train, X_val_test, y_train, y_val_test = train_test_split(
#         X, y, test_size=0.2, random_state=RANDOM_STATE
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_val_test, y_val_test, test_size=0.5, random_state=RANDOM_STATE
#     )

#     print(f'Обучающая выборка: {len(X_train)} записей')
#     print(f'Валидационная выборка: {len(X_val)} записей')
#     print(f'Тестовая выборка: {len(X_test)} записей')
    
#     return X_train, y_train, X_val, y_val, X_test, y_test
