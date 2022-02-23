
function run_hash() {
  echo "Running $1 bits"
  python hashing.py --train True --code_length "$1" --dataset "$2" --device "$3"
}

function run_attack() {
  echo "Running $1 bits"
  python attack.py --code_length "$1" --dataset "$2" --device "$3"  --method "$4"
}

dataset=$1
device=$2
method=$3

echo "Current dataset: $dataset"
run_attack 16 "$dataset" "$device" "$method"
run_attack 32 "$dataset" "$device" "$method"
run_attack 48 "$dataset" "$device" "$method"
run_attack 64 "$dataset" "$device" "$method"
