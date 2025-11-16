# --- hyperparams / config (edit as needed) ---
SEEDS=(40 41 42 43 44)
DATASET_NAME="IEMOCAP"
CLASS_NUM=6
BATCH_SIZE=40
ROBERTA_LR="2e-5"
ELSE_LR="1e-4"
HIDDEN_DIM=768
SPEAKER_STATE_DIM=328
TIME_DIM=0         # TIME_DIM=0なら「間」情報を使わない
HEADS=6
LOCAL_WINDOW_NUM=0
DROPOUT_RATE=0.1


# experiment name (used for log dirs)
EXP_NAME="robertaIr${ROBERTA_LR}_elseIr${ELSE_LR}_hiddenDim${HIDDEN_DIM}_speakerStateDim${SPEAKER_STATE_DIM}_timeDim${TIME_DIM}_head${HEADS}_localWindowNum${LOCAL_WINDOW_NUM}_dropout${DROPOUT_RATE}_AddPause"

# logs保存先
TEST_LOG_DIR="logs/test/${DATASET_NAME}/${EXP_NAME}"
mkdir -p "${TEST_LOG_DIR}"


for SEED in "${SEEDS[@]}"; do
    echo "========================== Testing with SEED=${SEED} =========================="
    
    # saved model filename (keeps same naming convention)
    TRAINED_SUBDIR="best_${EXP_NAME}"
    TRAINED_FILENAME="${TRAINED_SUBDIR}/seed${SEED}.pth"
    
    TEST_LOG_FILE="${TEST_LOG_DIR}/seed${SEED}.log"

    python -u test.py \
        --seed "${SEED}" \
        --dataset "${DATASET_NAME}" \
        --num_classes "${CLASS_NUM}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --speaker_state_dim "${SPEAKER_STATE_DIM}" \
        --time_dim "${TIME_DIM}" \
        --heads "${HEADS}" \
        --local_window_num "${LOCAL_WINDOW_NUM}" \
        --dropout_rate "${DROPOUT_RATE}" \
        --trained_filename "${TRAINED_FILENAME}" \
        2>&1 | tee "${TEST_LOG_FILE}"

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
done