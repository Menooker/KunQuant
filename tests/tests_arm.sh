set -e
echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant runtime tests"
python tests/test_runtime.py
echo "KunQuant alpha101 tests"
python tests/test_alpha101.py arm
#echo "KunQuant alpha158 tests"
#python ./tests/test_alpha158.py --inputs /tmp/input.npz --ref /tmp/alpha158.npz --action run_avx2
echo "KunQuant CAPI tests"
python ./tests/gen_alpha101_stream.py /tmp/
ls ./build/lib.linux-aarch64-cpython-39/KunQuant/runner/
LD_LIBRARY_PATH=./build/lib.linux-aarch64-cpython-39/KunQuant/runner/:${LD_LIBRARY_PATH} ./build/temp.linux-aarch64-cpython-39/KunCApiTest ./build/lib.linux-aarch64-cpython-39/KunQuant/runner/libKunTest.so /tmp/alpha101_stream/alpha101_stream.so
echo "All test done"