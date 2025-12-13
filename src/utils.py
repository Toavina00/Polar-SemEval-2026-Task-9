import csv
import os
import shutil
import zipfile

import torch
import pandas as pd
from transformers import DataCollatorWithPadding


def prepare_submission(dataset, model, thresh, batch_size, device):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(dataset.tokenizer)
    )

    outputs = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            ids = [dataset.ids[idx] for idx in batch["idx"]]
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            ).logits

            preds = (torch.sigmoid(logits) > thresh).int().cpu()

            for id, pred in zip(ids, preds):
                outputs.append([id] + pred.tolist())

    return outputs


def save_submission(filename, rows, header):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def compile_submission(
    save_name,
    root_dir,
    subtask_id,
    languages,
    model,
    thresh,
    tokenizer,
    batch_size,
    device
):
    subtask_dir = f"subtask_{subtask_id}"

    if os.path.exists(subtask_dir):
        shutil.rmtree(subtask_dir)
    os.makedirs(subtask_dir)

    for lang in languages:
        dev = pd.read_csv(os.path.join(root_dir, f'subtask{subtask_id}/dev/{lang}.csv'))

        dataset = PolarDataset(
            dev['id'].tolist(),
            dev['text'].tolist(),
            [],
            tokenizer,
            train=False
        )

        submission = prepare_submission(dataset, model, thresh, batch_size, device)
        labels = dev.columns.drop(["id", "text"]).tolist()

        pred_file = os.path.join(subtask_dir, f"pred_{lang}.csv")
        save_submission(pred_file, submission, ["id"] + labels)

    with zipfile.ZipFile(save_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(subtask_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(subtask_dir))
                zipf.write(file_path, arcname=arcname)