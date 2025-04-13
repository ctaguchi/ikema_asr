#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N ikema-asr-romaji
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

module load python
poetry run python finetune.py \
       --dataset ikema_youtube_asr \
       --script romaji \
       --wandb_run_name ikema-asr-youtube-romaji
    
cd -
