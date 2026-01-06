

# NODE_RANK=$1

export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=xxx
export WANDB_API_KEY=xxx

# Path of training data
DATA_PATH=YOUR_DATA_PATH
TOKENIZER_PATH=YOUR_TOKENIZER_PATH 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_SAMPLES=16
EPISODE=10000
WARMUP=0.0
TBS=256
RBS=64
KL=0.0
LR=2e-6
MAX_LENGTH=29000
PORT=1278
TEMP=1.0

SAVE_MODEL_NAME=xxx

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p /opt/aps/workdir/sht-RAG_RL/results/$SAVE_MODEL_NAME
mkdir -p /opt/aps/workdir/sht-RAG_RL/results/ckpts
mkdir -p /opt/aps/workdir/sht-RAG_RL/results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

ray job submit --address="http://127.0.0.1:8267" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path /opt/aps/workdir/sht-RAG_RL/results/ckpts/$SAVE_MODEL_NAME \
   --ckpt_path /opt/aps/workdir/sht-RAG_RL/results/ckpts/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator reinforce_baseline \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --load_checkpoint \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 3 \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
   # --apply_chat_template \
