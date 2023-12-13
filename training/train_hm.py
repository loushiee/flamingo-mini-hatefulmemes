"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import os
import random

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchvision.utils import save_image

from torchvision import transforms as T

import transformers
from transformers import HfArgumentParser, CLIPImageProcessor, EarlyStoppingCallback
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup

from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

from eval_hm import evaluate_hatefulness  # don't ask me why this import works
from hateful_memes_dataset import HatefulMemesDataset

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Hateful Memes dataset paths
HM_ROOT  = os.path.join('..', '..', 'hateful_memes')
HM_IMAGES = os.path.join(HM_ROOT, 'img') 
HM_ANN_TRAIN = os.path.join(HM_ROOT, 'train.jsonl') 
HM_ANN_SEEN_VAL = os.path.join(HM_ROOT, 'dev_seen.jsonl') 
HM_ANN_UNSEEN_VAL = os.path.join(HM_ROOT, 'dev_unseen.jsonl') 
HM_ANN_SEEN_TEST = os.path.join(HM_ROOT, 'test_seen.jsonl') 
HM_ANN_UNSEEN_TEST = os.path.join(HM_ROOT, 'test_unseen.jsonl') 
IMG_RESIZE = (224, 224)
IMG_MAX_SIZE = 360


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_type) # type: ignore

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values']

        
def prepare_training_dataset(config: FlamingoConfig):
    """ prepare a Hateful Memes training dataset """
    image_transform = T.Compose([
        T.RandomHorizontalFlip(),                       # add your favorite transforms
        #T.Resize(size=IMG_RESIZE, interpolation=T.InterpolationMode.BICUBIC),
        CLIPImageTransform(config.clip_model_type)
    ])

    def text_transform(text, label):
        prompt = HatefulMemesDataset.generate_prompt(text, label)
        return f"{random.choice(['', ' '])}{prompt}"

    return HatefulMemesDataset(
        HM_IMAGES, 
        HM_ANN_TRAIN, 
        image_transform=image_transform,
        text_transform=text_transform
    )
    

def prepare_evaluation_dataset(config: FlamingoConfig):
    image_transform = T.Compose([
        #T.Resize(size=IMG_RESIZE, interpolation=T.InterpolationMode.BICUBIC),
        CLIPImageTransform(config.clip_model_type)
    ])
    return HatefulMemesDataset(HM_IMAGES, HM_ANN_SEEN_VAL, annotations_path2=None,
        image_transform=image_transform)


def prepare_test_dataset(config: FlamingoConfig):
    image_transform = T.Compose([
        CLIPImageTransform(config.clip_model_type)
    ])
    return HatefulMemesDataset(HM_IMAGES, HM_ANN_SEEN_TEST, annotations_path2=None,
        image_transform=image_transform)


class DataCollator:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        images, texts, _ = zip(*batch)
        inputs = self.processor(text=texts)
        images = torch.stack(images)

        # rnd = random.getrandbits(10)
        # logging.info(f"DBGR: {rnd} DataCollator text: {texts[-1]}")
        # logging.info(f"DBGR: {rnd} DataCollator media_locations: {inputs['media_locations'][-1]}")
        # logging.info(f"DBGR: {rnd} DataCollator inputs_ids: {inputs['input_ids'][-1]}")
        # save_image(images[-1], f"/home/lou/Downloads/temp/DataCollator_{rnd}.png")

        return dict(
            pixel_values=images,
            labels=inputs['input_ids'],
            **inputs
        )


class FlamingoTrainer(Trainer):
    model: FlamingoModel
    eval_dataset: HatefulMemesDataset
    
    def evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """ override evaluation method to inject custom behavior. 
        TODO this only runs on one GPU, how to do distributed evaluation?
        """
        metrics = evaluate_hatefulness(self.eval_dataset, self.model, 
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            max_length=120,
        )
        metrics = {f"{metric_key_prefix}_{k}" : v for k, v in metrics.items()}

        # HF trainer stuff from overridden method
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    training_args: TrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s', 
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
    config = FlamingoConfig(
        clip_model_type='openai/clip-vit-large-patch14',
        lm='facebook/opt-350m',
        dim=1024,
        dim_visual=1024,
        xattn_act='sqrelu',
        resampler_act='sqrelu',
        resampler_depth=6,
        resampler_heads=8,
        resampler_num_latents=64,
        vocab_size=50273,
    )
    #model = FlamingoModel(config)
    model = FlamingoModel.from_pretrained('dhansmair/flamingo-mini')
    model.unfreeze_lm()
    model.unfreeze_vm()
    print(model)
    model.train()

    #################################################################
    # datasets
    #################################################################
    logger.info('loading datasets...')
    train_dataset = prepare_training_dataset(config)
    eval_dataset = prepare_evaluation_dataset(config)
    test_dataset = prepare_test_dataset(config)
    
    #################################################################
    # optimizer, scheduler, trainer
    #################################################################
    # optimizer = AdamW(model.parameters_trainable(), training_args.learning_rate)
    # scheduler = get_constant_schedule_with_warmup(optimizer, training_args.warmup_steps)

    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(config),
        callbacks=[EarlyStoppingCallback(10, 0.01)]
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    if training_args.resume_from_checkpoint is not None:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()

    logger.info('**** EVALUATE TEST SET ****')
    runs = 10
    acc_total = 0
    roc_auc_total = 0
    for i in range(runs):
        metrics = evaluate_hatefulness(test_dataset, model, 
            batch_size=64,
            num_workers=8,
            max_length=150,
        )
        acc_total += metrics["accuracy"]
        roc_auc_total += metrics["roc_auc"]

    logger.info(f'TEST ACCURACY: {acc_total / runs}')
    logger.info(f'TEST ROC AUC: {roc_auc_total / runs}')
        