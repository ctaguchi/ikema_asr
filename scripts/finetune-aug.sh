#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N ikema-asr-roma-aug
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"

module load python
poetry run python finetune.py \
       --dataset ikema_youtube_asr_aug \
       --load_from_disk \
       --eval_dataset ikema_youtube_asr_test \
       --epoch 30 \
       --script kana \
       --wandb_run_name ikema-asr-youtube-aug \
       --repo_name ikema-asr-youtube-aug
    
cd -
