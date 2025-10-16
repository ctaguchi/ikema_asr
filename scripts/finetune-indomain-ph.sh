#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N ikema-asr-indomain-ph
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"

module load python
poetry run python finetune.py \
       --dataset ikema_youtube_asr_full \
       --use_dict_dataset \
       --dict_dataset_path ikema_dict_asr \
       --phonemic_vocab \
       --epoch 100 \
       --batch_size 16 \
       --learning_rate 0.0003 \
       --script kana \
       --wandb_run_name ikema-asr-indomain-ph \
       --repo_name ikema-asr-indomain-ph
    
cd -
