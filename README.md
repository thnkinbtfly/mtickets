# Multilingual Lottery Tickets to Pretrain Language Models [EMNLP 2023 Findings]
This work is implemented based on [bert](https://github.com/google-research/bert) and [xtreme](https://github.com/google-research/xtreme).

## environments
We mainly used tpu-vms (tpu-v3-8, v2-alpha, tf2.6), and NGC 21.03 pytorch container (RTX 3090).
```
pip install -r requirements.txt
```

## preprocessing

After sentence splitting, run following:

```bash
export inputs=PREPROCESSED_DATA_DIR
export outputs=OUTPUT_DIR
export seq_len=128 # or 512
export max_preds=20 # or 80
export lang=LANGUAGE_CODE # refer to common.py
python create_pretraining_data.py --input_file=${inputs}/*  --vocab_file=mbert/vocab.txt --output_dir=${output_dir} --do_lower_case=False --max_seq_length=${seq_len} --random_seed 13370 --dupe_factor=5 --max_predictions_per_seq=${max_preds} --masked_lm_prob=0.15 --language ${lang}
```

Then generate randomly initialized bert checkpoint `random_init_bert`. You may use `convert_tf_to_torch.py` to generate pytorch(hf) equivalent.

## calculating SNIPs
```bash
export train_file=PREPROCESSED_TFRECORD_FILE
export model=RANDOMLY_INITIALIZED_MODEL
export want_to_see_toks=2500000
export snips_output_dir=SNIPS_OUTPUT_DIR

python get_subgraph/get_sg.py --train_file ${train_file} \
--model_name_or_path ${model} \
--masked_lang ${lang} \
--mask_method linear \
--want_to_see_token_nums ${want_to_see_toks} \
--output_dir $snips_output_dir
```

## find multilingual tickets
```bash
exporl tickets_output_dir=TICKETS_OUTPUT_DIR
exporl tickets_output_dir_on_gs=TICKETS_OUTPUT_DIR_ON_GS
export remain_ratio=0.75 # ~ 12/16. or 0.85 ~ 12/14.
python get_subgraph/aggregate_snips.py --snips_folder $snips_output_dir --model_name_or_path ${model} \
--remain_ratio ${remain_ratio} --output_dir ${tickets_output_dir}
python convert_torch_to_tf.py --model_name=${tickets_output_dir} --tf_cache_dir=${tickets_output_dir_tf} --ckpt_save_name=model.ckpt
gsutil -m cp -r $tickets_output_dir_tf/\* ${tickets_output_dir_tf_on_gs}/
```
if you want to try random ticktes, add `--use_random_snips`, when running `aggregate_snips.py`.

## Pretrain on TPUv3-8 with multilingual tickets
```bash
export PYTHONPATH=`pwd`
export seed=42
export TPU_NAME=local
export MAX_SEQ=128
export MAX_STEPS=1000000
export MID_STEPS=900000
export WARMUP_STEPS=10000
export SAVE_STEPS=100000
export ITER_LOOP=5000
export MAX_PRED=20
export BATCH_SIZE=256
export lr=1e-4
export method=linear # or constant, if you don't use any masking (naive multilingual pretraining)
export input_file=INPUT_CSV_SEQ_128 # every row: lang_tfrecord_dir,sample_ratio
export output_dir=PRETRAIN_OUTPUT_DIR # note: all dir tpu use must be on gs bucket
export init_ckpt=${tickets_output_dir_tf_on_gs}/model.ckpt
export bert_config_json=mbert/bert_config.json # or modified version, if you scale the model

export COMMON_TRAIN_SETTINGS="--seed=${seed} --output_dir=${output_dir} --train_batch_size=${BATCH_SIZE}   --num_warmup_steps=${WARMUP_STEPS}  --total_train_steps=${MAX_STEPS}  --learning_rate=${lr}   --log_on_tb_steps 50 --keep_checkpoint_max 2 --do_train=True  "
export MODEL_SETTINGS="--bert_config_file=$bert_config_json"
export MASK_SETTINGS="--init_checkpoint=${init_ckpt} --opt_weight_only=True --attention_mask_method=${method} --value_attention_mask_method=${method} --output_mask_method=${method} --intermediate_mask_method=${method} --layer_mask_method=${method} --pool_mask_method=${method} "
export TPU_SETTINGS="--use_tpu=True   --tpu_name=$TPU_NAME"
export TRAIN_SETTINGS="--input_file_csv=${input_file}  --max_seq_length=${MAX_SEQ}   --max_predictions_per_seq=${MAX_PRED}   --save_checkpoints_steps=${SAVE_STEPS} --iterations_per_loop=${ITER_LOOP} --num_train_steps=${MID_STEPS} "

python run_pretraining.py $MASK_SETTINGS $COMMON_TRAIN_SETTINGS $TPU_SETTINGS $DATA_SETTINGS $TRAIN_SETTINGS $MODEL_SETTINGS

## 512 * 100K

export MAX_SEQ=512
export MAX_STEPS=1000000
export WARMUP_STEPS=10000
export SAVE_STEPS=25000
export ITER_LOOP=1000
export MAX_PRED=80
export input_file=INPUT_CSV_SEQ_512

export TRAIN_SETTINGS="--input_file_csv=${input_file} --max_seq_length=${MAX_SEQ}   --max_predictions_per_seq=${MAX_PRED}   --save_checkpoints_steps=${SAVE_STEPS} --iterations_per_loop=${ITER_LOOP} --num_train_steps=${MAX_STEPS} "

python run_pretraining.py $MASK_SETTINGS $COMMON_TRAIN_SETTINGS $TPU_SETTINGS $DATA_SETTINGS $TRAIN_SETTINGS $MODEL_SETTINGS

## weight prepare
export ckpt_down_dir=CKPT_DOWN_DIR
export hf_save_dir=HF_SAVE_DIR

mkdir -p $ckpt_down_dir
mkdir -p $hf_save_dir

gsutil -m cp -r ${output_dir}/model.ckpt\* ${ckpt_down_dir}/
gsutil -m cp -r ${output_dir}/checkpoint ${ckpt_down_dir}/
python convert_tf_to_torch.py --tf_checkpoint ${ckpt_down_dir}/model.ckpt-${MAX_STEPS} --config $bert_config_json --pytorch_dump_output ${hf_save_dir}/pytorch_model.bin --hf_config_path=mbert --with_mask
```

## Fine-tune on GPU server to evaluate
```bash
cd xtreme
bash install_tools.sh
export DATA_DIR="download/"
bash scripts/download_data.sh
export MODEL_CLS="bert-base-multilingual-cased"
bash $REPO/scripts/preprocess_udpos.sh $MODEL_CLS $DATA_DIR
bash $REPO/scripts/preprocess_panx.sh $MODEL_CLS $DATA_DIR

## NER
export PYTHONPATH=`pwd`
export MODEL=${hf_save_dir}
export LANGS=LANGUAGE_CODE
export MODEL_TYPE=bert_with_mask
export OUT_DIR="outputs/"
export MAX_LENGTH=128
export NUM_EPOCHS=2
export LR=2e-5
export BATCH_SIZE=8
export GRAD_ACC=4
export mask_method=linear # or constant

export TASK='panx'
export DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
export OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}_${LANGS}"
mkdir -p $OUTPUT_DIR
python third_party/run_tag.py \
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
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --adaptive_least_seen 10000 \
  --overwrite_output_dir \
  --save_only_best_checkpoint --mask_method=${mask_method}

## POS
export SAVE_STEPS=500 # 4000 if LANGUAGE_CODE is de, since it has much more data -> takes too long time
export TASK='udpos'
export DATA_DIR="download/"
export DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
export OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}_${LANGS}"
    mkdir -p $OUTPUT_DIR
    python third_party/run_tag.py \
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
      --learning_rate $LR \
      --do_train \
      --do_predict \
      --predict_langs $LANGS \
      --train_langs $LANGS \
      --log_file $OUTPUT_DIR/train.log \
      --adaptive_least_seen 10000 \
      --overwrite_output_dir --mask_method=${mask_method} --do_eval --eval_all_checkpoints --save_only_best_checkpoint 

```
## Check gc or uc
### getting uc
1. To get uc, use `--accum_type updates` to the pretraining script. To constrain the memory usage, we recommend to add `--keep_var layer_xx `, where xx is 0,1,...,11.
2. `python util/check_vec_change.py --ckpt_dir=$PRETRAIN_OUTPUT_DIR --output_dir=OUTPUT_DIR_FOR_UC --save_cossim `

### getting gc
1. First, generate some checkpoints with the "getting uc" script, modifying `--keep_checkpoint_max` and `--save_checkpoints_steps`. After train finishes, you may use `util/checkgrad_prepare.py`
2. Starting from each checkpoint, use `--accum_type assign_grad` to save gradients. Should be faster to collect all languages if you change the sample_ratio in input_csv as equiprobable.
3. Iterate through all `ORIG_SAVED_STEP` and run `python util/check_gradients.py --output_dir=OUTPUT_DIR_FOR_GC --ckpt_dirs=${PRETRAIN_OUTPUT_DIR}model.ckpt-${ORIG_SAVED_STEP} --train_steps=$save_checkpoints_steps`.
4. Watch log_names.txt to make sure that gradients from every language and components are written
