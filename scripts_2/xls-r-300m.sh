#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N xls-r-300m-ikema-kana
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"
REPO_NAME="xls-r-300m-ikema-kana"
RUN_NAME="xls-r-300m-ikema-kana"
MODEL="facebook/wav2vec2-xls-r-300m"

poetry run python finetune.py \
       --dataset ikema_youtube_asr_full \
       --eval_dataset ikema_youtube_asr_test \
       --model ${MODEL} \
       --epoch 30 \
       --script kana \
       --wandb_run_name ${RUN_NAME} \
       --repo_name ${REPO_NAME}
    
cd -
