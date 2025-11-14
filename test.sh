# --- hyperparams / config (edit as needed) ---
SEED=44
DATASET_NAME="IEMOCAP"
CLASS_NUM=6
BATCH_SIZE=40
ROBERTA_LR="2e-5"
ELSE_LR="1e-4"
HIDDEN_DIM=384
EMOTION_DIM=64
PAUSE_DIM=0         # PAUSE_DIM=0なら「間」情報を使わない
HEADS=6
LOCAL_WINDOW_NUM=15
DROPOUT_RATE=0.1


# experiment name (used for log dirs)
EXP_NAME="roberta_lr${ROBERTA_LR}_else_lr${ELSE_LR}_hidden_dim${HIDDEN_DIM}_emotion_dim${EMOTION_DIM}_pauseDim${PAUSE_DIM}_head${HEADS}_localWindowNum${LOCAL_WINDOW_NUM}_dropout${DROPOUT_RATE}"

# saved model filename (keeps same naming convention)
TRAINED_SUBDIR="best_${EXP_NAME}"
TRAINED_FILENAME="${TRAINED_SUBDIR}/seed${SEED}.pth"

# logs保存先
TEST_LOG_DIR="logs/test/${DATASET_NAME}/${EXP_NAME}"
mkdir -p "${TEST_LOG_DIR}"
TEST_LOG_FILE="${TEST_LOG_DIR}/seed${SEED}.log"


python -u test.py \
    --seed "${SEED}" \
    --dataset "${DATASET_NAME}" \
    --num_classes "${CLASS_NUM}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --emotion_dim "${EMOTION_DIM}" \
    --pause_dim "${PAUSE_DIM}" \
    --heads "${HEADS}" \
    --local_window_num "${LOCAL_WINDOW_NUM}" \
    --dropout_rate "${DROPOUT_RATE}" \
    --trained_filename "${TRAINED_FILENAME}" \
    2>&1 | tee "${TEST_LOG_FILE}"