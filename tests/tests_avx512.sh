set -e
echo "KunQuant compiler tests"
echo "KunQuant alpha101 tests"
sde -clx -env PYTHONPATH "$PYTHONPATH" -- python tests/test_alpha101.py avx512
echo "KunQuant alpha158 tests"
sde -clx -env PYTHONPATH "$PYTHONPATH" -- python ./tests/test_alpha158.py --avx512 --inputs /tmp/input.npz --ref /tmp/alpha158.npz
echo "All test done"