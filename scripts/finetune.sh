#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N ikema-asr-romaji
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"

module load python
poetry run python finetune.py \
       --dataset ikema_youtube_asr \
       --eval_dataset ikema_youtube_asr_test \
       --epoch 150 \
       --script romaji \
       --wandb_run_name ikema-asr-youtube-romaji \
       --repo_name ikema-asr-youtube-romaji
    
cd -
