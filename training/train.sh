#!/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--output_dir ./flamingo-hm
--run_name flamingo-mini-vitL
--do_train --do_eval
--optim adamw_torch
--lr_scheduler_type constant_with_warmup
--weight_decay 0.7
--learning_rate 0.00002
--warmup_steps 2000
--per_device_train_batch_size 16
--per_device_eval_batch_size 64
--gradient_accumulation_steps 1
--evaluation_strategy epoch
--save_strategy epoch
--save_total_limit 5
--log_level info
--dataloader_num_workers 8
--dataloader_pin_memory True
--fp16
--ddp_find_unused_parameters False
--num_train_epochs 50
--metric_for_best_model eval_accuracy
--load_best_model_at_end True
"

#--eval_steps 1000

echo $ARGS

if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python ./train_hm.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU ./train.py $ARGS
fi
