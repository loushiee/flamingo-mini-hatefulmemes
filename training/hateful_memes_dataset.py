import json
import logging
import os
import random

from torch.utils.data import Dataset
# from torchvision.utils import save_image
from PIL import Image

class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path, annotations_path2=None, image_transform=None, text_transform=None):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]
        
        # Add the 2nd set of annotations if valid
        if annotations_path2 is not None:
            with open(annotations_path2, "r") as f:
                self.annotations.extend([json.loads(line) for line in f])

        self.ids = [ann["id"] for ann in self.annotations]
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.dbgr_limit = 10

        logging.info(f"DBGR HatefulMemesDataset annotations size: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        text = annotation["text"]
        label = annotation["label"]

        # Apply transforms to image and text
        if self.image_transform:
            image = self.image_transform(image)
            # filename = annotation["img"].split("/")[-1]
            # save_image(image, f"/home/lou/Downloads/temp/{filename}")
        if self.text_transform:
            text = self.text_transform(text, label)

        return image, text, label
    
    @staticmethod
    def generate_prompt(text, label=None):
        answer = 'yes' if label == 1 else 'no'
        prompt = f'<image>is an image with written: "{text}" on it. Is it hateful? Answer:{answer if label is not None else ""}{"<EOC></s>" if label is not None else ""}'
        # logging.info(f"DBGR PROMPT: {prompt}")
        return prompt
