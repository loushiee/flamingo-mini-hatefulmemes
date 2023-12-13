
import logging
from dataclasses import dataclass, field
import os

import torch
from torchvision import transforms as T

import transformers
from transformers import AutoModel, HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup

from flamingo_mini import FlamingoConfig, FlamingoModel

from eval_hm import evaluate_hatefulness  # don't ask me why this import works
from hateful_memes_dataset import HatefulMemesDataset

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hateful Memes dataset paths
HM_ROOT  = os.path.join('..', '..', 'hateful_memes')
HM_IMAGES = os.path.join(HM_ROOT, 'img') 
HM_ANN_SEEN_TEST = os.path.join(HM_ROOT, 'test_seen.jsonl') 
HM_ANN_UNSEEN_TEST = os.path.join(HM_ROOT, 'test_unseen.jsonl') 


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_type) # type: ignore

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values']


def prepare_test_dataset():
    image_transform = T.Compose([
        CLIPImageTransform('openai/clip-vit-large-patch14')
    ])
    return HatefulMemesDataset(HM_IMAGES, HM_ANN_SEEN_TEST, annotations_path2=None,
        image_transform=image_transform)

    
if __name__ == '__main__':
    checkpoint_folder = './flamingo-hm/save'

    logging.basicConfig(
        format=f'%(asctime)s {checkpoint_folder} %(message)s', 
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ]    
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Loading pretrained model from '{checkpoint_folder}'...")
    model = FlamingoModel.from_pretrained(checkpoint_folder)

    device = torch.device("cuda")
    model.to(device)
    print(model)

    model.eval()

    logger.info('**** LOAD TEST DATA SET ****')
    test_dataset = prepare_test_dataset()

    logger.info('**** EVALUATE TEST SET ****')
    runs = 1
    acc_total = 0
    roc_auc_total = 0
    for i in range(runs):
        metrics = evaluate_hatefulness(test_dataset, model, 
            batch_size=64,
            num_workers=8,
            max_length=150,
        )
        print(metrics)
        acc_total += metrics["accuracy"]
        roc_auc_total += metrics["roc_auc"]

    logger.info(f'TEST ACCURACY: {acc_total / runs}')
    logger.info(f'TEST ROC AUC: {roc_auc_total / runs}')
