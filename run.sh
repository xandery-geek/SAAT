
function run_hash() {
  echo "Running $1 bits"
  python hashing.py --train True --code_length $1 --dataset $2 --device $3
}

function run_hag() {
    echo "Running $1 bits"
    python attack.py --method hag --code_length $1 --dataset $2 --device $3
}

function run_sdha() {
    echo "Running $1 bits"
    python attack.py --method sdha --code_length $1 --dataset $2 --device $3
}

dataset=$1
device=$2


echo "Current dataset: $dataset"
run_hash 16 "$dataset" "$device"
run_hash 32 "$dataset" "$device"
run_hash 48 "$dataset" "$device"
run_hash 64 "$dataset" "$device"
