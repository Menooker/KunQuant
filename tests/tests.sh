set -e
echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant runtime tests"
python tests/test_runtime.py
echo "KunQuant alpha101 tests"
python tests/test_alpha101.py avx2
# echo "KunQuant alpha158 tests"
# python ./tests/test_alpha158.py --inputs /tmp/input.npz --ref /tmp/alpha158.npz 
echo "KunQuant CAPI tests"
python ./tests/gen_alpha101_stream.py /tmp/
./build/temp.linux-x86_64-3.9/KunCApiTest ./build/lib.linux-x86_64-cpython-39/KunQuant/runner/libKunTest.so /tmp/alpha101_stream/alpha101_stream.so
echo "KunQuant AVX512 alpha101 tests"
sde -clx -env PYTHONPATH "$PYTHONPATH" -- python tests/test_alpha101.py avx512
# echo "KunQuant AVX512 alpha158 tests"
# sde -clx -env PYTHONPATH "$PYTHONPATH" -- python ./tests/test_alpha158.py --avx512 --inputs /tmp/input.npz --ref /tmp/alpha158.npz
echo "All test done"