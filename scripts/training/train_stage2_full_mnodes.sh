#!/usr/bin/env bash
set -euo pipefail

# Stage-2 2D/3D training. All paths and launch settings can be overridden via environment variables.
NNODES=${1:-${NNODES:-1}}
LOCAL_BS=${2:-${LOCAL_BS:-10}}
LOCAL_BS_3D=${3:-${LOCAL_BS_3D:-2}}


SCRIPT_DIR="$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
WORKSPACE="$(cd -P -- "${SCRIPT_DIR}/../.." && pwd -P)"

echo "💡💡💡 WORKSPACE: ${WORKSPACE}"

cd "${WORKSPACE}"

DATASET_TYPE=${DATASET_TYPE:-csvimgfeat}
TRAIN_DATA=${TRAIN_DATA:-"${WORKSPACE}/data/examples_2d.csv"}
TRAIN_DATA_3D=${TRAIN_DATA_3D:-"${WORKSPACE}/data/examples_3d.csv"}
TRAIN_ROOT=${TRAIN_ROOT:-"${WORKSPACE}/data"}
FEATURE_ROOT=${FEATURE_ROOT:-"${WORKSPACE}/outputs/alignment"}
CSV_IMG_KEY=${CSV_IMG_KEY:-identifier}
CSV_CAPTION_KEY=${CSV_CAPTION_KEY:-submodality}
MIXED_RES=${MIXED_RES:-True}
CONTRAST=${CONTRAST:-clip}
INIT_CKPT=${INIT_CKPT:-"${WORKSPACE}/weights/ckpt.pth"}

WORKERS=${WORKERS:-10}
EPOCH=${EPOCH:-1}
MODEL=${MODEL:-medvtok-clip-meanpool}
LR=${LR:-2e-5}
LSHARE=${LSHARE:-1}
LEXPERT=${LEXPERT:-1}
LSLIDR=${LSLIDR:-0.1}

EXP_NAME=${EXP_NAME:-"DAT-per100v6_NET-${MODEL}-mproj_LOS-rec-gan-kl-${CONTRAST}-lshare-${LSHARE}-lexpert-${LEXPERT}-lslidr-${LSLIDR}_LR-${LR}_ND-${NNODES}"}
RESUME_NET_ONLY=${RESUME_NET_ONLY:-True}  # do not load optimizer
OUTPUT_DIR=${OUTPUT_DIR:-"${WORKSPACE}/outputs/ckpts/${EXP_NAME}"}
RESUME_FROM=${RESUME_FROM:-}
CORE_WEIGHTS_ONLY=${CORE_WEIGHTS_ONLY:-False}
PADDING_MODE=${PADDING_MODE:-zero}

IMG_SIZE=${IMG_SIZE:-256}
NUM_APPEAR_EACH_MODALITY=${NUM_APPEAR_EACH_MODALITY:-2}
REPORT_WANDB=${REPORT_WANDB:-False}
EVAL_PER_EPOCH=${EVAL_PER_EPOCH:-4}
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
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

torchrun --nnodes="${NNODES}" --nproc-per-node="${NGPUS}" --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" --node-rank="${NODE_RANK}" \
--module main \
--bcr 0 \
--epoch "${EPOCH}" --lr "${LR}" --lshare "${LSHARE}" --lexpert "${LEXPERT}" \
--lslidr "${LSLIDR}" \
--num_appear_each_modality "${NUM_APPEAR_EACH_MODALITY}" \
--padding_mode "${PADDING_MODE}" \
--init_ckpt_path "${INIT_CKPT}" \
--local_bs "${LOCAL_BS}" \
--local_bs_3d "${LOCAL_BS_3D}" \
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
--train_data_3d "${TRAIN_DATA_3D}" \
--train_root_3d "${TRAIN_ROOT}" \
--include_3d_projector True \
--include_3d_data True \
--workers "${WORKERS}" \
--vis_img_dir "${VIS_IMG_DIR}" \
--output_dir "${OUTPUT_DIR}" \
--resume_from "$RESUME_FROM" \
--resume_net_only "${RESUME_NET_ONLY}" \
--core_weights_only "${CORE_WEIGHTS_ONLY}" \
--img_size "${IMG_SIZE}" \
--enable_visualizer "${ENABLE_VISUALIZER}" \
--mixed_resolution_training "${MIXED_RES}" \
--contrastive_type "${CONTRAST}"
