import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, ids, texts, labels, tokenizer, train=True, max_length=256):
        self.ids = ids
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length
        )

        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item['idx'] = idx

        if self.train:
            label = self.labels[idx]
            dtype = torch.float if isinstance(label, list) else torch.long
            item['labels'] = torch.tensor(label, dtype=dtype)

        return item


def load_data(data_dir, languages):
    dfs = []
    for lang in languages:
        path = os.path.join(data_dir, f"{lang}.csv")
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def compute_weights(dataset):
    if not dataset.train or len(dataset.labels) == 0:
        return None
    
    labels_array = torch.tensor(dataset.labels)
    num_labels = labels_array.shape[1]
    weights = []
    
    for i in range(num_labels):
        pos = labels_array[:, i].sum().item()
        neg = len(labels_array) - pos
        weights.append(neg / pos if pos > 0 else 1.0)
    
    return torch.tensor(weights)


def create_datasets(df, labels, tokenizer, train, val_size, random_state, stratify):
    if val_size > 0 and train:
        stratify_key = None
        if stratify:
            stratify_key = df[labels].apply(lambda x: ''.join(x.astype(str)), axis=1)
        
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_key
        )
        
        train_dataset = PolarDataset(
            train_df['id'].tolist(),
            train_df['text'].tolist(),
            train_df[labels].values.tolist(),
            tokenizer,
            train=True
        )
        
        val_dataset = PolarDataset(
            val_df['id'].tolist(),
            val_df['text'].tolist(),
            val_df[labels].values.tolist(),
            tokenizer,
            train=True
        )
        
        return train_dataset, val_dataset
    
    dataset = PolarDataset(
        df['id'].tolist(),
        df['text'].tolist(),
        [],
        tokenizer,
        train=False
    )
    
    return dataset


def load_dataset(
    data_dir,
    languages,
    tokenizer,
    train=True,
    val_size=0.2,
    random_state=42,
    stratify=False
):
    df = load_data(data_dir, languages)
    labels = df.columns.drop(["id", "text"]).tolist()
    
    datasets = create_datasets(df, labels, tokenizer, train, val_size, random_state, stratify)
    
    result = {"labels": labels}
    
    if isinstance(datasets, tuple):
        result["train_dataset"] = datasets[0]
        result["val_dataset"] = datasets[1]
        result["weights"] = compute_weights(datasets[0])
    else:
        result["dataset"] = datasets
        if is_train:
            result["weights"] = compute_weights(datasets)
    
    return result