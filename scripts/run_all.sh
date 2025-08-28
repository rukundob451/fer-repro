#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

# preprocess genome -> arch & target
python src/process_pb.py --zip_path picbreeder_genomes/skull.zip --save_dir data/skull_pb

# baseline
python src/train_sgd.py \
  --arch "12;cache:15,gaussian:4,identity:2,sin:1" \
  --img_file data/skull_pb/img.png \
  --n_iters 30000 --lr 1e-3 --init_scale 0.1 \
  --lambda_sym 0.0 \
  --save_dir data/sgd_skull_lambda0 > logs/lambda0.log 2>&1

# lambda variants
for L in 0.05 0.2 0.5; do
  tag=${L/./}
  python src/train_sgd.py \
    --arch "12;cache:15,gaussian:4,identity:2,sin:1" \
    --img_file data/skull_pb/img.png \
    --n_iters 30000 --lr 1e-3 --init_scale 0.1 \
    --lambda_sym ${L} \
    --save_dir data/sgd_skull_lambda${tag} > logs/lambda${tag}.log 2>&1
done
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"

mkdir -p "$ROOT/logs"

# preprocess genome -> arch & target
$PY src/process_pb.py --zip_path picbreeder_genomes/skull.zip --save_dir data/skull_pb

# baseline
$PY src/train_sgd.py \
  --arch "12;cache:15,gaussian:4,identity:2,sin:1" \
  --img_file data/skull_pb/img.png \
  --n_iters 30000 --lr 1e-3 --init_scale 0.1 \
  --lambda_sym 0.0 \
  --save_dir data/sgd_skull_lambda0 > logs/lambda0.log 2>&1

# lambda variants
for L in 0.05 0.2 0.5; do
  tag=${L/./}
  $PY src/train_sgd.py \
    --arch "12;cache:15,gaussian:4,identity:2,sin:1" \
    --img_file data/skull_pb/img.png \
    --n_iters 30000 --lr 1e-3 --init_scale 0.1 \
    --lambda_sym ${L} \
    --save_dir data/sgd_skull_lambda${tag} > logs/lambda${tag}.log 2>&1
done
