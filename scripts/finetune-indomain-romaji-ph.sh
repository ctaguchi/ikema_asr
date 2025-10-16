#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N ikema-asr-indomain-romaji-ph
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"
REPO_NAME="ikema-asr-indomain-romaji-ph"

module load python
poetry run python finetune.py \
       --dataset ikema_youtube_asr_full \
       --use_dict_dataset \
       --dict_dataset_path ikema_dict_asr \
       --phonemic_vocab \
       --epoch 100 \
       --batch_size 16 \
       --learning_rate 0.0003 \
       --script romaji \
       --wandb_run_name ${REPO_NAME} \
       --repo_name ${REPO_NAME}
    
cd -
