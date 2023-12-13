from typing import Optional, List, Dict
from tqdm import tqdm

import logging
import math
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.utils import save_image

from flamingo_mini import FlamingoModel, FlamingoProcessor
from hateful_memes_dataset import HatefulMemesDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MyDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, text, label = self.dataset[index]
        image_id = self.dataset.ids[index]
        prompt = HatefulMemesDataset.generate_prompt(text)
        gt = HatefulMemesDataset.generate_prompt(text, label)
        return image, image_id, prompt, gt, label


@torch.no_grad()
def evaluate_hatefulness(
    dataset: HatefulMemesDataset,
    model: FlamingoModel, 
    *,
    max_length: int = 80,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    
    wrapper = MyDatasetWrapper(dataset)
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers)

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0
    count = 0

    for pixels, pixel_ids, prompts, gts, labels in tqdm(loader, disable=not verbose):
        loss, preds, gens, log_probs = model.predict_memes_hatefulness(
            processor, 
            pixel_values=pixels.to(model.device),
            prompts=prompts,
            gts=gts,
            max_length=max_length,
        )

        all_labels.extend(labels)
        all_preds.extend(preds)
        total_loss += loss

        limit = 16
        raw_probs= []
        for g, l, p, img, img_id, prob in zip(gens, labels, preds, pixels, pixel_ids, log_probs):
            all_probs.append(round(prob, 3))

            # if count == 0 and limit > 0:
            #     logging.info(f"DBGR {'*PASS*' if l == p else '*FAIL*'} ID:{(img_id)} PRED:{'YES' if p == 1 else 'NO'} GNRTD:'{g}'")
            #     save_image(img, f"/home/lou/Downloads/temp/{img_id}.png")
            #     limit -= 1

        # if count == 0:
        #     logging.info(f"DBGR labels:\n{labels}\npreds:{preds}\nall_probs:\n{all_probs}")
            
        count += 1


    metrics = {}
    metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    metrics["F1"] = f1_score(all_labels, all_preds)
    metrics["loss"] = total_loss / count
    return metrics

