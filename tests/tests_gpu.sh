set -e
echo "KunQuant runtime tests"
python tests/test_runtime.py --gpu-arch auto
echo "KunQuant alpha158 tests"
python ./tests/test_alpha158.py --inputs ./build/input.npz --ref ./build/alpha158.npz --action run_gpu --gpu-arch auto
echo "KunQuant alpha101 tests"
python ./tests/test_alpha101.py --gpu-arch auto