

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT=${DATA_ROOT:-"$ROOT_DIR/../data"}
HEALTHY_ROOT=${HEALTHY_DATA_ROOT:-"$ROOT_DIR/../data/healthy_ct"}
LIST_ROOT=${LIST_ROOT:-"$ROOT_DIR/cross_eval/liver_aug_data_fold"}

python generate_tumor.py \
    --data_root "$DATA_ROOT" \
    --healthy_data_root "$HEALTHY_ROOT" \
    --data_list "$LIST_ROOT/real_tumor_train_0.txt" \
    --output_dir ./generated_tumors \
    --organ_type liver \
    --tumor_type all \
    --num_samples 1
