#!/usr/bin/env bash
#
# Train, then run inference, for every TorchWM model -- one model after another.
#
# Each model is a pair of stages: a training run scaled by --preset, followed by
# a recorded inference pass that picks up the checkpoint that training just
# wrote. A stage that fails is logged and the sweep moves on, so one broken
# model does not hide the state of the others. A summary table prints at the end
# and the exit status is non-zero if any stage failed.
#
# This is the end-to-end counterpart to scripts/benchmark_models.sh, which times
# forward/backward passes and never trains.
#
# Usage:
#   scripts/run_all_models.sh                       # tiny train+infer, every model
#   scripts/run_all_models.sh --preset small
#   scripts/run_all_models.sh --models diamond,iris --preset paper --device cuda
#   scripts/run_all_models.sh --train-only
#   scripts/run_all_models.sh --infer-only          # reuse existing checkpoints
#   scripts/run_all_models.sh --list
#   scripts/run_all_models.sh --dry-run --preset paper
#
# Presets:
#   tiny   A couple of epochs at batch 2. Minutes on CPU. Proves the loop runs
#          end to end; it learns nothing. This is the default.
#   small  The repo's single-GPU configs (iris_small_gpu.yaml,
#          jepa_small_gpu.yaml) and equivalent scale-downs elsewhere.
#   paper  The published configs, untouched. Days of GPU time per model.
#
# tiny and small change scale only -- batch sizes, step counts, epochs. Nothing
# that defines a method (token counts, masking geometry, imagination horizons,
# loss weights) is touched at any preset, so --preset paper reproduces the
# published configuration exactly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_PY="${SCRIPT_DIR}/benchmark_infer.py"
EXP_DIR="${REPO_ROOT}/torchwm/configs/experiments"

# Models with both a training entrypoint and a recorded-inference demo.
ALL_MODELS="diamond dreamer iris genie dit jepa"
# Train-only extras: no inference demo exists for these.
EXTRA_MODELS="planet rssm world-model"

PRESET="tiny"
MODELS=""
INCLUDE_EXTRA=0
LIST_ONLY=0
RUN_TRAIN=1
RUN_INFER=1
DEVICE=""
STEPS="120"
SEED="0"
TIMEOUT="0"
DRY_RUN=0
FAIL_FAST=0
DO_SYNC=1
USE_UV=1
PYTHON_BIN="${PYTHON:-python}"
PYTHON_VERSION=""
SYNC_EXTRAS=()
CKPT_ROOT="${REPO_ROOT}/checkpoints"
OUT_DIR="${REPO_ROOT}/results/model_runs"
JEPA_DATA="${TORCHWM_JEPA_DATA:-${IMAGENET_ROOT:-}}"
GENIE_DATASET=""
GENIE_DATA_FILE=""
WM_ENV="Pendulum-v1"

usage() {
    cat <<'USAGE_EOF'
Usage: scripts/run_all_models.sh [flags]

Selection:
  --models a,b,c       Only these models (default: diamond,dreamer,iris,genie,dit,jepa).
  --all                Also run the train-only extras: planet, rssm, world-model.
  --preset tiny|small|paper
                       Scale of the training runs (default: tiny).
  --list               Show what would run, then exit.

Stages:
  --train-only         Train, skip inference.
  --infer-only         Skip training; use whatever checkpoints already exist.

Run control:
  --device NAME        cpu / cuda / cuda:0. Passed to every model that takes one.
  --steps N            Inference frames to record per model (default: 120).
  --seed N             Seed for training and inference (default: 0).
  --timeout SECONDS    Kill any stage that runs longer (0 = no limit). Needs
                       coreutils timeout. Useful for planet/rssm, which have no
                       CLI knobs to shorten them.
  --fail-fast          Stop at the first failing stage (default: keep going).
  --dry-run            Print every command instead of running it.

Data (models whose training needs a dataset):
  --jepa-data PATH     Image folder for I-JEPA. Also read from TORCHWM_JEPA_DATA
                       or IMAGENET_ROOT. Without it, JEPA training is skipped.
  --genie-dataset NAME TinyWorlds dataset for Genie (e.g. SONIC). Without it,
                       Genie training only validates trainer construction.
  --genie-data-file PATH
                       Local TinyWorlds HDF5 file; avoids the download.
  --wm-env NAME        Gym env for the world-model trainer (default: Pendulum-v1).

Output:
  --ckpt-root PATH     Where checkpoints are written (default: ./checkpoints).
  --out-dir PATH       Where videos and logs land (default: ./results/model_runs).

Environment:
  --no-sync            Skip uv sync; use the environment as-is.
  --no-uv              Do not use uv at all; run with $PYTHON (default: python).
  --extra NAME         Add an optional dependency group to the sync (repeatable).
  --python VERSION     Interpreter for the uv environment, e.g. --python 3.12.
  -h, --help           This message.

Examples:
  scripts/run_all_models.sh --preset tiny --device cpu
  scripts/run_all_models.sh --models dreamer --preset small --device cuda
  scripts/run_all_models.sh --infer-only --models diamond,iris
  scripts/run_all_models.sh --all --timeout 300 --dry-run
USAGE_EOF
}

die() { echo "error: $*" >&2; exit 2; }
need_value() { [ "$2" -ge 2 ] || die "$1 needs a value"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models)          need_value "$1" "$#"; MODELS="$2"; shift 2 ;;
        --models=*)        MODELS="${1#*=}"; shift ;;
        --preset)          need_value "$1" "$#"; PRESET="$2"; shift 2 ;;
        --preset=*)        PRESET="${1#*=}"; shift ;;
        --all)             INCLUDE_EXTRA=1; shift ;;
        --list)            LIST_ONLY=1; shift ;;
        --train-only)      RUN_INFER=0; shift ;;
        --infer-only)      RUN_TRAIN=0; shift ;;
        --device)          need_value "$1" "$#"; DEVICE="$2"; shift 2 ;;
        --device=*)        DEVICE="${1#*=}"; shift ;;
        --steps)           need_value "$1" "$#"; STEPS="$2"; shift 2 ;;
        --steps=*)         STEPS="${1#*=}"; shift ;;
        --seed)            need_value "$1" "$#"; SEED="$2"; shift 2 ;;
        --seed=*)          SEED="${1#*=}"; shift ;;
        --timeout)         need_value "$1" "$#"; TIMEOUT="$2"; shift 2 ;;
        --timeout=*)       TIMEOUT="${1#*=}"; shift ;;
        --fail-fast)       FAIL_FAST=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --jepa-data)       need_value "$1" "$#"; JEPA_DATA="$2"; shift 2 ;;
        --jepa-data=*)     JEPA_DATA="${1#*=}"; shift ;;
        --genie-dataset)   need_value "$1" "$#"; GENIE_DATASET="$2"; shift 2 ;;
        --genie-dataset=*) GENIE_DATASET="${1#*=}"; shift ;;
        --genie-data-file) need_value "$1" "$#"; GENIE_DATA_FILE="$2"; shift 2 ;;
        --genie-data-file=*) GENIE_DATA_FILE="${1#*=}"; shift ;;
        --wm-env)          need_value "$1" "$#"; WM_ENV="$2"; shift 2 ;;
        --wm-env=*)        WM_ENV="${1#*=}"; shift ;;
        --ckpt-root)       need_value "$1" "$#"; CKPT_ROOT="$2"; shift 2 ;;
        --ckpt-root=*)     CKPT_ROOT="${1#*=}"; shift ;;
        --out-dir)         need_value "$1" "$#"; OUT_DIR="$2"; shift 2 ;;
        --out-dir=*)       OUT_DIR="${1#*=}"; shift ;;
        --no-sync)         DO_SYNC=0; shift ;;
        --no-uv)           USE_UV=0; DO_SYNC=0; shift ;;
        --extra)           need_value "$1" "$#"; SYNC_EXTRAS+=("--extra" "$2"); shift 2 ;;
        --extra=*)         SYNC_EXTRAS+=("--extra" "${1#*=}"); shift ;;
        --python)          need_value "$1" "$#"; PYTHON_VERSION="$2"; shift 2 ;;
        --python=*)        PYTHON_VERSION="${1#*=}"; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 die "unknown flag '$1' (try --help)" ;;
    esac
done

case "${PRESET}" in
    tiny|small|paper) ;;
    *) die "unknown preset '${PRESET}' (tiny, small or paper)" ;;
esac

if [ "${RUN_TRAIN}" -eq 0 ] && [ "${RUN_INFER}" -eq 0 ]; then
    die "--train-only and --infer-only cancel out"
fi

# ---------------------------------------------------------------- model table

selected_models() {
    if [ -n "${MODELS}" ]; then
        echo "${MODELS}" | tr ',' ' '
    elif [ "${INCLUDE_EXTRA}" -eq 1 ]; then
        echo "${ALL_MODELS} ${EXTRA_MODELS}"
    else
        echo "${ALL_MODELS}"
    fi
}

known_model() {
    case " ${ALL_MODELS} ${EXTRA_MODELS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# DiT is the only model here with no training entrypoint of its own.
can_train() { [ "$1" != "dit" ]; }

can_infer() {
    case " ${ALL_MODELS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# The env a model is trained on at this preset. Inference has to replay the same
# one -- a Pendulum policy cannot be rolled out in walker-walk.
model_env() {
    case "$1" in
        diamond) echo "Breakout-v5" ;;
        iris)    echo "ALE/Pong-v5" ;;
        dreamer)
            if [ "${PRESET}" = "paper" ]; then echo "walker-walk"; else echo "Pendulum-v1"; fi
            ;;
        *) echo "" ;;
    esac
}

# ------------------------------------------------------------ command builders
#
# Each builder fills CMD. A non-empty SKIP_REASON means the stage cannot run
# here and is reported as SKIP rather than FAIL. NOTE is a caveat printed
# alongside the result without changing it.

CMD=()
SKIP_REASON=""
NOTE=""

build_train_cmd() {
    local model="$1"
    CMD=(); SKIP_REASON=""; NOTE=""

    case "${model}" in
        diamond)
            CMD=("${RUNNER[@]}" -m torchwm.training.train_diamond "seed=${SEED}")
            case "${PRESET}" in
                tiny)
                    CMD+=(preset=small num_epochs=2 training_steps_per_epoch=2
                          environment_steps_per_epoch=64 batch_size=2
                          num_sampling_steps=1 use_amp=false
                          data_loader_num_workers=0 pin_memory=false
                          persistent_workers=false
                          save_interval=1 eval_interval=1 log_interval=1)
                    ;;
                small)
                    CMD+=(--config "${EXP_DIR}/diamond.yaml"
                          preset=small num_epochs=50 training_steps_per_epoch=100
                          environment_steps_per_epoch=100 batch_size=8
                          save_interval=10 eval_interval=10)
                    ;;
                paper)
                    CMD+=(--config "${EXP_DIR}/diamond.yaml")
                    ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            ;;

        dreamer)
            CMD=("${RUNNER[@]}" -m torchwm.training.train_dreamer
                 "logdir=${CKPT_ROOT}/dreamer" "seed=${SEED}")
            case "${PRESET}" in
                tiny)
                    CMD+=(env_backend=gym env=Pendulum-v1
                          total_steps=600 seed_steps=200 collect_steps=200
                          update_steps=2 batch_size=4 train_seq_len=10
                          max_episode_length=100 time_limit=100
                          checkpoint_interval=200 test_interval=1000000
                          scalar_freq=100 log_video_freq=-1)
                    ;;
                small)
                    CMD+=(env_backend=gym env=Pendulum-v1
                          total_steps=100000 seed_steps=5000 batch_size=16
                          train_seq_len=25 checkpoint_interval=10000)
                    ;;
                paper)
                    # Config defaults are the paper's: dmc walker-walk, 5M steps.
                    ;;
            esac
            # Dreamer selects its device with a flag, not a device string.
            [ "${DEVICE}" = "cpu" ] && CMD+=(no_gpu=true)
            ;;

        iris)
            CMD=("${RUNNER[@]}" -m torchwm.training.train_iris
                 "save_dir=${CKPT_ROOT}/iris" "seed=${SEED}")
            case "${PRESET}" in
                tiny)
                    # Scale levers only. tokens_per_frame, imagination_horizon,
                    # burn_in_length and every loss weight stay at paper values.
                    CMD+=(epochs=2 collection_epochs=1 env_steps_per_epoch=64
                          training_steps_per_epoch=2 transformer_steps_per_epoch=2
                          actor_critic_steps_per_epoch=2
                          autoencoder_batch_size=2 transformer_batch_size=2
                          actor_critic_batch_size=2
                          start_autoencoder_after=0 start_transformer_after=0
                          start_actor_critic_after=0
                          checkpoint_interval=1 eval_episodes=1
                          max_env_steps=128 use_amp=false)
                    ;;
                small)
                    CMD+=(--config "${EXP_DIR}/iris_small_gpu.yaml")
                    ;;
                paper)
                    CMD+=(--config "${EXP_DIR}/iris.yaml")
                    ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            ;;

        genie)
            if [ -n "${GENIE_DATASET}" ]; then
                CMD=("${RUNNER[@]}" "${SCRIPT_DIR}/train_genie_tinyworlds.py"
                     "dataset=${GENIE_DATASET}"
                     "checkpoint_dir=${CKPT_ROOT}/genie")
                case "${PRESET}" in
                    tiny)
                        CMD+=(max_steps=20 batch_size=1 num_frames=8
                              num_workers=0 log_interval=5 val_interval=1000000)
                        ;;
                    small) CMD+=(max_steps=5000 batch_size=2 log_interval=100) ;;
                    paper) CMD+=(max_steps=50000) ;;
                esac
                if [ -n "${GENIE_DATA_FILE}" ]; then
                    CMD+=("data_file=${GENIE_DATA_FILE}")
                else
                    NOTE="downloads the TinyWorlds dataset"
                fi
                [ -n "${DEVICE}" ] && CMD+=("device=${DEVICE}")
            else
                # No dataset: validate that the trainer builds, and stop there.
                CMD=("${RUNNER[@]}" -m torchwm.training.train_genie --dry-run)
                [ "${PRESET}" = "tiny" ] && CMD+=(--max-steps 20)
                [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")
                NOTE="trainer construction only; pass --genie-dataset to train"
            fi
            ;;

        jepa)
            if [ -z "${JEPA_DATA}" ]; then
                SKIP_REASON="no image dataset (--jepa-data PATH, TORCHWM_JEPA_DATA or IMAGENET_ROOT)"
                return
            fi
            CMD=("${RUNNER[@]}" -m torchwm.training.train_jepa)
            case "${PRESET}" in
                tiny)
                    CMD+=(meta.model_name=vit_tiny meta.use_bfloat16=false
                          data.batch_size=2 data.num_workers=0
                          optimization.epochs=1 optimization.warmup=0)
                    NOTE="one epoch over ${JEPA_DATA}; bound it with --timeout"
                    ;;
                small) CMD+=(--config "${EXP_DIR}/jepa_small_gpu.yaml") ;;
                paper) CMD+=(--config "${EXP_DIR}/jepa.yaml") ;;
            esac
            # After --config so these win over the file.
            CMD+=("data.root_path=${JEPA_DATA}" "logging.folder=${CKPT_ROOT}/jepa")
            ;;

        planet|rssm)
            CMD=("${RUNNER[@]}" -m "torchwm.training.train_${model}")
            NOTE="no CLI knobs; length is fixed in the module -- use --timeout"
            ;;

        world-model)
            CMD=("${RUNNER[@]}" -m torchwm.training.train_world_model
                 --env "${WM_ENV}"
                 --logdir "${CKPT_ROOT}/world_model"
                 --data_dir "${CKPT_ROOT}/world_model/data")
            case "${PRESET}" in
                tiny)
                    CMD+=(--num_rollouts 4 --vae_epochs 1 --rnn_epochs 1
                          --vae_batch_size 4 --rnn_batch_size 2 --seq_len 8
                          --ctrl_pop_size 2 --ctrl_samples 1 --ctrl_workers 1
                          --ctrl_time_limit 100 --stage vae)
                    NOTE="VAE stage only"
                    ;;
                small)
                    CMD+=(--num_rollouts 100 --vae_epochs 10 --rnn_epochs 10
                          --ctrl_workers 2)
                    ;;
                paper) ;;
            esac
            [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")
            ;;

        dit)
            SKIP_REASON="no training entrypoint (inference-only model)"
            ;;

        *)
            SKIP_REASON="unknown model"
            ;;
    esac
}

# Newest existing file among the arguments, or empty.
newest_match() {
    local newest="" candidate
    for candidate in "$@"; do
        [ -e "${candidate}" ] || continue
        if [ -z "${newest}" ] || [ "${candidate}" -nt "${newest}" ]; then
            newest="${candidate}"
        fi
    done
    echo "${newest}"
}

find_checkpoint() {
    local model="$1" found=""
    # Unmatched globs must expand to nothing rather than to themselves.
    shopt -s nullglob
    case "${model}" in
        diamond)
            found="$(newest_match "${CKPT_ROOT}"/diamond/checkpoint_*.pt \
                                  "${CKPT_ROOT}"/diamond/*.pt)"
            ;;
        iris)
            found="$(newest_match "${CKPT_ROOT}"/iris/final_*.pt \
                                  "${CKPT_ROOT}"/iris/best_*.pt \
                                  "${CKPT_ROOT}"/iris/checkpoint_*.pt)"
            ;;
        dreamer)
            found="$(newest_match "${CKPT_ROOT}"/dreamer/ckpts/*_ckpt.pt \
                                  "${CKPT_ROOT}"/dreamer/*/ckpts/*_ckpt.pt)"
            ;;
        genie)
            found="$(newest_match "${CKPT_ROOT}"/genie/*.pt)"
            ;;
        jepa)
            found="$(newest_match "${CKPT_ROOT}"/jepa/*.pth.tar \
                                  "${CKPT_ROOT}"/jepa/*.pt)"
            ;;
    esac
    shopt -u nullglob
    echo "${found}"
}

build_infer_cmd() {
    local model="$1"
    CMD=(); SKIP_REASON=""; NOTE=""

    local checkpoint env
    checkpoint="$(find_checkpoint "${model}")"

    CMD=("${RUNNER[@]}" "${INFER_PY}" --mode record --model "${model}"
         --out-dir "${OUT_DIR}/videos" --steps "${STEPS}" --seed "${SEED}")
    [ -n "${DEVICE}" ] && CMD+=(--device "${DEVICE}")

    env="$(model_env "${model}")"
    [ -n "${env}" ] && CMD+=(--game "${env}")

    case "${model}" in
        diamond|dreamer|iris)
            if [ -z "${checkpoint}" ]; then
                SKIP_REASON="no checkpoint under ${CKPT_ROOT}/${model} (train first)"
                return
            fi
            CMD+=(--checkpoint "${checkpoint}")
            [ "${model}" = "iris" ] && CMD+=(--episodes 1)
            [ "${model}" = "diamond" ] && CMD+=(--dream-steps "${STEPS}")
            ;;
        genie|dit|jepa)
            # These demos run without weights, so an untrained sweep still
            # exercises the whole generation path.
            if [ -n "${checkpoint}" ]; then
                CMD+=(--checkpoint "${checkpoint}")
            else
                CMD+=(--random-init)
                NOTE="random init (no checkpoint found)"
            fi
            ;;
        *)
            SKIP_REASON="no inference demo for this model"
            ;;
    esac
}

# --------------------------------------------------------------------- listing

MODEL_LIST="$(selected_models)"
for model in ${MODEL_LIST}; do
    known_model "${model}" || \
        die "unknown model '${model}'. Known: ${ALL_MODELS} ${EXTRA_MODELS}"
done

if [ "${LIST_ONLY}" -eq 1 ]; then
    stages=""
    [ "${RUN_TRAIN}" -eq 1 ] && stages="train"
    [ "${RUN_INFER}" -eq 1 ] && stages="${stages}${stages:+ }infer"
    echo "preset: ${PRESET}    stages: ${stages}"
    echo
    printf '%-14s %-7s %-7s %s\n' "MODEL" "TRAIN" "INFER" "NOTES"
    for model in ${MODEL_LIST}; do
        trainable="yes"; inferable="yes"; notes=""
        can_train "${model}" || { trainable="no"; notes="inference only"; }
        can_infer "${model}" || { inferable="no"; notes="train only"; }
        case "${model}" in
            jepa)  [ -z "${JEPA_DATA}" ] && notes="needs --jepa-data" ;;
            genie) [ -z "${GENIE_DATASET}" ] && notes="dry-run without --genie-dataset" ;;
            planet|rssm) notes="fixed length; use --timeout" ;;
        esac
        printf '%-14s %-7s %-7s %s\n' "${model}" "${trainable}" "${inferable}" "${notes}"
    done
    exit 0
fi

# ----------------------------------------------------------------- environment

ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0

    # A previous install may be on disk but not on PATH.
    local candidate
    for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
        if [ -x "${candidate}" ]; then
            PATH="$(dirname "${candidate}"):${PATH}"
            export PATH
            return 0
        fi
    done

    echo "uv not found -- installing it from https://astral.sh/uv ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "need curl or wget to install uv, or install it yourself and re-run with --no-uv"
    fi

    PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    export PATH
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after installing"
}

cd "${REPO_ROOT}"

if [ "${USE_UV}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ]; then
    ensure_uv
    echo "uv: $(uv --version)"
fi

if [ "${DO_SYNC}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ]; then
    # Training and recording both need Gymnasium and OpenCV; pull those in
    # unless the caller already named the group.
    extras_flat=" ${SYNC_EXTRAS[*]-} "
    case "${extras_flat}" in *" gym "*) ;; *) SYNC_EXTRAS+=("--extra" "gym") ;; esac
    case "${extras_flat}" in *" viz "*) ;; *) SYNC_EXTRAS+=("--extra" "viz") ;; esac

    # --inexact installs what the lock requires without uninstalling anything
    # else already in the environment.
    sync_cmd=(uv sync --inexact)
    [ -n "${PYTHON_VERSION}" ] && sync_cmd+=(--python "${PYTHON_VERSION}")
    sync_cmd+=("${SYNC_EXTRAS[@]}")
    echo "installing dependencies: ${sync_cmd[*]}"
    "${sync_cmd[@]}"
fi

if [ "${USE_UV}" -eq 1 ]; then
    # --no-sync: the environment is already resolved above.
    RUNNER=(uv run --no-sync python -u)
else
    RUNNER=("${PYTHON_BIN}" -u)
fi

LOG_DIR="${OUT_DIR}/logs"
if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${LOG_DIR}" "${OUT_DIR}/videos" "${CKPT_ROOT}"
fi

if [ "${TIMEOUT}" != "0" ] && ! command -v timeout >/dev/null 2>&1; then
    echo "warning: --timeout ${TIMEOUT} ignored -- no 'timeout' on PATH" >&2
    TIMEOUT="0"
fi

# ------------------------------------------------------------------- execution

ROWS=()
FAILURES=0

record_row() {
    # model | stage | status | duration | detail
    ROWS+=("$1|$2|$3|$4|$5")
}

run_stage() {
    local model="$1" stage="$2"
    local log="${LOG_DIR}/${model}.${stage}.log"

    if [ "${stage}" = "train" ]; then
        build_train_cmd "${model}"
    else
        build_infer_cmd "${model}"
    fi

    if [ -n "${SKIP_REASON}" ]; then
        echo ">> ${model} / ${stage}: SKIP -- ${SKIP_REASON}"
        record_row "${model}" "${stage}" "SKIP" "-" "${SKIP_REASON}"
        return 0
    fi

    local launch=("${CMD[@]}")
    if [ "${TIMEOUT}" != "0" ]; then
        launch=(timeout --preserve-status -k 10 "${TIMEOUT}" "${CMD[@]}")
    fi

    echo
    echo "===================================================================="
    echo ">> ${model} / ${stage} (preset ${PRESET})"
    [ -n "${NOTE}" ] && echo "   note: ${NOTE}"
    echo "   ${launch[*]}"
    echo "===================================================================="

    if [ "${DRY_RUN}" -eq 1 ]; then
        record_row "${model}" "${stage}" "DRY" "-" "${NOTE}"
        return 0
    fi

    local start=${SECONDS} status=0
    set +e
    "${launch[@]}" 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    set -e
    local elapsed=$((SECONDS - start))

    if [ "${status}" -eq 0 ]; then
        record_row "${model}" "${stage}" "OK" "${elapsed}s" "${NOTE}"
        return 0
    fi

    FAILURES=$((FAILURES + 1))
    record_row "${model}" "${stage}" "FAIL(${status})" "${elapsed}s" "log: ${log}"
    echo ">> ${model} / ${stage} FAILED (exit ${status}); see ${log}" >&2
    [ "${FAIL_FAST}" -eq 1 ] && return 1
    return 0
}

echo "models:      ${MODEL_LIST}"
echo "preset:      ${PRESET}"
echo "device:      ${DEVICE:-auto}"
echo "checkpoints: ${CKPT_ROOT}"
echo "output:      ${OUT_DIR}"

for model in ${MODEL_LIST}; do
    ran_any=0
    if [ "${RUN_TRAIN}" -eq 1 ] && can_train "${model}"; then
        ran_any=1
        run_stage "${model}" "train" || break
    fi
    if [ "${RUN_INFER}" -eq 1 ] && can_infer "${model}"; then
        ran_any=1
        run_stage "${model}" "infer" || break
    fi
    if [ "${ran_any}" -eq 0 ]; then
        # Asked for a stage this model does not have, e.g. --train-only dit.
        echo ">> ${model}: nothing to do for the selected stage(s)"
        record_row "${model}" "-" "SKIP" "-" "no such stage for this model"
    fi
done

# --------------------------------------------------------------------- summary

echo
echo "============================== summary ============================="
printf '%-14s %-6s %-10s %-8s %s\n' "MODEL" "STAGE" "STATUS" "TIME" "DETAIL"
for row in ${ROWS[@]+"${ROWS[@]}"}; do
    IFS='|' read -r r_model r_stage r_status r_time r_detail <<<"${row}"
    printf '%-14s %-6s %-10s %-8s %s\n' \
        "${r_model}" "${r_stage}" "${r_status}" "${r_time}" "${r_detail}"
done
echo "===================================================================="

if [ "${DRY_RUN}" -eq 0 ]; then
    echo "logs:        ${LOG_DIR}"
    echo "videos:      ${OUT_DIR}/videos"
    echo "checkpoints: ${CKPT_ROOT}"
fi

if [ "${FAILURES}" -gt 0 ]; then
    echo
    echo "${FAILURES} stage(s) failed." >&2
    exit 1
fi
