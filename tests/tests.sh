set -e
echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant runtime tests"
python tests/test_runtime.py
echo "KunQuant alpha101 tests"
python tests/test_alpha101.py
echo "KunQuant alpha158 tests"
python python ./tests/test_alpha158.py --inputs /tmp/input.npz --ref /tmp/alpha158.npz 
echo "KunQuant CAPI tests"
./build/KunCApiTest ./build/libKunTest.so ./build/projects/libAlpha101Stream.so 
echo "All test done"