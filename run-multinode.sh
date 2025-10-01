# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=16
ACCOUNT=coreai_dlalgo_llm
JOB_NAME=coreai_dlalgo_llm:nemotron-sft
PARTITION=batch

export WANDB_API_KEY=""
export HF_TOKEN=""
export HF_HOME="/lustre/fsw/portfolios/coreai/users/shashankv/hf_home"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/coreai/users/shashankv/hf_home/datasets"

COMMAND="NRL_FORCE_REBUILD_VENVS=true NCCL_NVLS_ENABLE=1 uv run examples/run_sft.py --config examples/configs/sft_nemotron_super_49b_tulu_v3.yaml policy.max_total_sequence_length=32768 policy.train_global_batch_size=128 sft.val_global_batch_size=128 policy.dtensor_cfg.context_parallel_size=8 policy.dtensor_cfg.tensor_parallel_size=4 ++data.cache_dir=${HF_HOME} cluster.num_nodes=${NUM_ACTOR_NODES}" \
CONTAINER="/lustre/fsw/portfolios/coreai/users/shashankv/containers/shashankv-nemo-rl-2509rc2.sqsh" \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
