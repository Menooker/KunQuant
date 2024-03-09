set -e
echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant runtime tests"
python tests/test_runtime.py
echo "KunQuant alpha101 tests"
python tests/test_alpha101.py
echo "KunQuant CAPI tests"
./build/KunCApiTest ./build/libKunTest.so 
echo "All test done"