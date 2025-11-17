# Use $2 if provided, otherwise use environment variable CUDA_VISIBLE_DEVICES
# Set memory optimization: expandable_segments to reduce fragmentation
CUDA_VISIBLE_DEVICES=${2:-${CUDA_VISIBLE_DEVICES:-0}} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MAX_PIXELS=400000 \
  swift sft \
  --round 30 \
  --round_per_epoch 10 \
  --fed_alg central \
  --client_num 1 \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path /home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct \
  --lazy_tokenize True \
  --preprocess_num_proc 4 \
  --dataset "/home/hmpiao/hmpiao/xuerong/FedMABench/android_control_unpack/episode-wise-conversations.jsonl" \
  --sft_type lora \
  --tuner_backend peft \
  --dtype AUTO \
  --output_dir output \
  --train_dataset_sample -1 \
  --dataset_test_ratio 0 \
  --max_steps -1 \
  --max_length 2048 \
  --check_dataset_strategy warning \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing true \
  --batch_size 1 \
  --weight_decay 0.1 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.03 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --save_only_model True \
  --logging_steps 100

#  --custom_train_dataset_path /GPFS/data/wenhaowang-1/ms-swift/androidcontrol_1108/unpack-1109-test-message-vlm-train.jsonl \
