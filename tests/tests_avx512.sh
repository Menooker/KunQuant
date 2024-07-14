set -e
echo "KunQuant compiler tests"
echo "KunQuant alpha101 tests"
sde -clx -env PYTHONPATH "$PYTHONPATH" -- python tests/test_alpha101.py
echo "KunQuant alpha158 tests"
sde -clx -env PYTHONPATH "$PYTHONPATH" -- python ./tests/test_alpha158.py --inputs /tmp/input.npz --ref /tmp/alpha158.npz
echo "All test done"