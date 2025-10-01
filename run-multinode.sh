# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=16
ACCOUNT=coreai_dlalgo_llm
JOB_NAME=coreai_dlalgo_llm:nemotron-sft
PARTITION=batch

export WANDB_API_KEY=""
export HF_TOKEN=""
export HF_HOME="/lustre/fsw/portfolios/coreai/users/shashankv/hf_home"
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/coreai/users/shashankv/hf_home/datasets"
export RESULTS_DIR="/lustre/fsw/portfolios/coreai/users/shashankv/results/sft_nemotron_49b"

COMMAND="NRL_FORCE_REBUILD_VENVS=true NCCL_NVLS_ENABLE=1 uv run examples/run_sft.py --config examples/configs/sft_nemotron_super_49b_tulu_v3.yaml ++data.cache_dir=${HF_HOME} cluster.num_nodes=${NUM_ACTOR_NODES} checkpointing.checkpoint_dir=${RESULTS_DIR}" \
CONTAINER="/lustre/fsw/portfolios/coreai/users/shashankv/containers/shashankv-nemo-rl-2509rc2.sqsh" \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub