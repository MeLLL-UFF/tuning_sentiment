#!/bin/bash

###################### Downloads necessários #############################

pip3 install tensorflow==1.12.0
#python -c 'import tensorflow as tf; print(tf.__version__)'
git clone https://github.com/pytorch/fairseq.git 
pip3 install --editable ./fairseq
pip3 install fairseq

mkdir -p roberta.base
curl https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz -o roberta.base/roberta.base.tar.gz
tar -xzvf roberta.base/roberta.base.tar.gz

PATH_dir="/home/sergio_barreto/sentiment-embeddings/projeto-modularizar/inputs"
INPUT_PATH=${PATH_dir}"/dataset_consolidado_ppbase.csv"

###################### Split dataset ##################################
pip3 install sklearn
mkdir -p ${PATH_dir}/raw_dataset_finetuning
python3 Split_dataset.py --input_path $INPUT_PATH

################## encoding dataset with the GPT-2 BPE #################

mkdir -p gpt2_bpe
curl https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json -o gpt2_bpe/encoder.json 
curl https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe -o gpt2_bpe/vocab.bpe 
curl https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt -o gpt2_bpe/dict.txt
mkdir -p ${PATH_dir}/BPE
for SPLIT in train val test; do \
    echo ${SPLIT}
    python3 ./fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs ${PATH_dir}/raw_dataset_finetuning/dataset.${SPLIT}.csv \
        --outputs ${PATH_dir}/BPE/dataset.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done


######### preprocess/binarize the data using the GPT-2 fairseq dictionary ################

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ${PATH_dir}/BPE/dataset.train.bpe \
    --validpref ${PATH_dir}/BPE/dataset.val.bpe \
    --testpref ${PATH_dir}/BPE/dataset.test.bpe \
    --destdir data-bin/dataset \
    --workers 60

################ Train RoBERTa base ################

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/dataset

fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --restore-file ./roberta.base/model.pt


python3 Test_model.py
