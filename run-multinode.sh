# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=8
ACCOUNT=coreai_dlalgo_genai
JOB_NAME=coreai_dlalgo_genai:nemotron-sft
PARTITION=batch

export WANDB_API_KEY=""
export HF_TOKEN=""

COMMAND="NRL_FORCE_REBUILD_VENVS=true NCCL_NVLS_ENABLE=1   uv run --extra automodel examples/run_sft.py --config examples/configs/sft_tulu_v3.yaml cluster.num_nodes=${NUM_ACTOR_NODES}" \
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