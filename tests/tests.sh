set -e
echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant runtime tests"
python tests/test_runtime.py
echo "KunQuant alpha101 tests"
python tests/test_alpha101.py
echo "KunQuant alpha158 tests"
python ./tests/test_alpha158.py --inputs /tmp/input.npz --ref /tmp/alpha158.npz 
echo "KunQuant CAPI tests"
python ./tests/gen_alpha101_stream.py /tmp/
./build/temp.linux-x86_64-3.9/KunCApiTest ./build/lib.linux-x86_64-3.9/KunQuant/runner/libKunTest.so /tmp/alpha101_stream/alpha101_stream.so
echo "All test done"