#!/usr/bin/env bash
set -euo pipefail

# Stage-1 2D-only training. Paths and launch settings can be overridden via
# environment variables. The first two positional arguments remain compatible
# with the original script: NNODES and per-device batch size.
NNODES=${1:-${NNODES:-1}}
LOCAL_BS=${2:-${LOCAL_BS:-10}}

SCRIPT_DIR="$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
WORKSPACE="$(cd -P -- "${SCRIPT_DIR}/../.." && pwd -P)"

echo "💡💡💡 WORKSPACE: ${WORKSPACE}"
cd "${WORKSPACE}"

DATASET_TYPE=${DATASET_TYPE:-csvimgfeat}
TRAIN_DATA=${TRAIN_DATA:-"${WORKSPACE}/data/examples_2d.csv"}
TRAIN_ROOT=${TRAIN_ROOT:-"${WORKSPACE}/data"}
FEATURE_ROOT=${FEATURE_ROOT:-"${WORKSPACE}/outputs/alignment"}
CSV_IMG_KEY=${CSV_IMG_KEY:-identifier}
CSV_CAPTION_KEY=${CSV_CAPTION_KEY:-submodality}
MIXED_RES=${MIXED_RES:-True}
CONTRAST=${CONTRAST:-clip}


WORKERS=${WORKERS:-16}
EPOCH=${EPOCH:-1}
MODEL=${MODEL:-medvtok-clip-meanpool}
LR=${LR:-1e-4}
LSHARE=${LSHARE:-1}
LEXPERT=${LEXPERT:-1}
LSLIDR=0

EXP_NAME=${EXP_NAME:-"DAT-per100v6_NET-${MODEL}-mproj_LOS-rec-gan-kl-${CONTRAST}-lshare-${LSHARE}-lexpert-${LEXPERT}_LR-${LR}_ND-${NNODES}"}
RESUME_NET_ONLY=${RESUME_NET_ONLY:-True}
OUTPUT_DIR=${OUTPUT_DIR:-"${WORKSPACE}/outputs/ckpts/${EXP_NAME}"}
RESUME_FROM=${RESUME_FROM:-}
SIGLIP_ENCODER_INIT=${SIGLIP_ENCODER_INIT:-}
CORE_WEIGHTS_ONLY=${CORE_WEIGHTS_ONLY:-False}
PADDING_MODE=${PADDING_MODE:-zero}

IMG_SIZE=${IMG_SIZE:-256}
NUM_APPEAR_EACH_MODALITY=${NUM_APPEAR_EACH_MODALITY:-2}
REPORT_WANDB=${REPORT_WANDB:-False}
EVAL_PER_EPOCH=${EVAL_PER_EPOCH:-8}
ENABLE_VISUALIZER=${ENABLE_VISUALIZER:-True}
VIS_IMG_DIR=${VIS_IMG_DIR:-"${WORKSPACE}/data/vis_imgs"}


for required_dir in "${TRAIN_ROOT}" "${FEATURE_ROOT}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Required directory not found: ${required_dir}" >&2
        exit 2
    fi
done

NGPUS=${NPROC_PER_NODE:-1}
echo "GPUs per node: ${NGPUS}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-20165}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

torchrun --nnodes="${NNODES}" --nproc-per-node="${NGPUS}" \
--master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
--node-rank="${NODE_RANK}" \
--module main \
--epoch "${EPOCH}" --lr "${LR}" --lshare "${LSHARE}" --lexpert "${LEXPERT}" \
--lslidr "${LSLIDR}" \
--num_appear_each_modality "${NUM_APPEAR_EACH_MODALITY}" \
--padding_mode "${PADDING_MODE}" \
--local_bs "${LOCAL_BS}" \
--report_wandb "${REPORT_WANDB}" \
--eval_per_epoch "${EVAL_PER_EPOCH}" \
--model "${MODEL}" \
--exp_name "${EXP_NAME}" \
--dataset_type "${DATASET_TYPE}" \
--csv_img_key "${CSV_IMG_KEY}" \
--csv_caption_key "${CSV_CAPTION_KEY}" \
--feature_root "${FEATURE_ROOT}" \
--train_root "${TRAIN_ROOT}" \
--train_data "${TRAIN_DATA}" \
--include_3d_projector False \
--include_3d_data False \
--workers "${WORKERS}" \
--vis_img_dir "${VIS_IMG_DIR}" \
--output_dir "${OUTPUT_DIR}" \
--resume_from "${RESUME_FROM}" \
--siglip_encoder_init "${SIGLIP_ENCODER_INIT}" \
--resume_net_only "${RESUME_NET_ONLY}" \
--core_weights_only "${CORE_WEIGHTS_ONLY}" \
--img_size "${IMG_SIZE}" \
--enable_visualizer "${ENABLE_VISUALIZER}" \
--mixed_resolution_training "${MIXED_RES}" \
--contrastive_type "${CONTRAST}"
