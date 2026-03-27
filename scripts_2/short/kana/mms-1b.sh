x#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N mms-1b-ikema-kana
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

cd ../../../src/

export HF_HOME="/afs/crc.nd.edu/group/nlp/07/ctaguchi/.cache/"
export WANDB_PROJECT="ikema_asr"
REPO_NAME="mms-1b-ikema-kana"
RUN_NAME="mms-1b-ikema-kana"
MODEL="facebook/mms-1b"
OUTPUT_DIR="../models/${REPO_NAME}"

module load python
poetry run python finetune.py \
       --model ${MODEL} \
       --epoch 10 \
       --script kana \
       --wandb_run_name ${RUN_NAME} \
       --repo_name ${REPO_NAME} \
       --output_dir ${OUTPUT_DIR}
    
cd -
