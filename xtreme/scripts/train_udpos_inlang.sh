#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
LANGS=${2:-"en"}
MODEL_TYPE=${3:-"bert"}
SAVE_STEPS=${4-"500"}
ADDITIONAL_PARAM=${5:-""}
ADDITIONAL_PARAM2=${6:-""}
DATA_DIR="$REPO/download/"
OUT_DIR="$REPO/outputs/"

TASK='udpos'
NUM_EPOCHS=2
MAX_LENGTH=128
LR=2e-5

LC=""
MODEL_CLS="bert-base-multilingual-cased"
if [ $LANGS == "kk" ]; then
  EVAL_SETTINGS=""
else
  EVAL_SETTINGS="--do_eval --eval_all_checkpoints --save_only_best_checkpoint "
fi

#if [ $MODEL_CLS == "bert-base-multilingual-cased" ]; then
#  MODEL_TYPE="bert"
#elif [ $MODEL_CLS == "xlm-mlm-100-1280" ] || [ $MODEL_CLS == "xlm-mlm-tlm-xnli15-1024" ]; then
#  MODEL_TYPE="xlm"
#  LC=" --do_lower_case"
#elif [ $MODEL_CLS == "xlm-roberta-large" ] || [ $MODEL_CLS == "xlm-roberta-base" ]; then
#  MODEL_TYPE="xlmr"
#fi

if [ $MODEL_CLS == "xlm-mlm-100-1280" ] || [ $MODEL_CLS == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}${ADDITIONAL_PARAM}${ADDITIONAL_PARAM2}_${LANGS}"
mkdir -p $OUTPUT_DIR
python $REPO/third_party/run_tag.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --tokenizer_name $MODEL_CLS \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed 1 \
  --learning_rate $LR \
  --do_train \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --adaptive_least_seen 10000 \
  --overwrite_output_dir \
  $EVAL_SETTINGS $LC $ADDITIONAL_PARAM  $ADDITIONAL_PARAM2
rm -r $OUTPUT_DIR
