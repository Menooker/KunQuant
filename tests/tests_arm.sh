set -e

OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
    SYSTEM_TRIPPLE="linux-aarch64"
    DYN_LIB_EXT="so"
elif [ "$OS" = "Darwin" ]; then
    SYSTEM_TRIPPLE="macosx-10.9-universal2"
    DYN_LIB_EXT="dylib"
else
    SYSTEM_TRIPPLE="$OS"
fi
echo "SYSTEM_TRIPPLE=${SYSTEM_TRIPPLE}"

echo "KunQuant compiler tests"
python tests/test.py
python tests/test2.py
echo "KunQuant alpha158 tests"
python ./tests/test_alpha158.py --action run


if [ "$OS" = "Linux" ]; then
    export LD_LIBRARY_PATH=./build/lib.${SYSTEM_TRIPPLE}-cpython-39/KunQuant/runner/:${LD_LIBRARY_PATH}
elif [ "$OS" = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=./build/lib.${SYSTEM_TRIPPLE}-cpython-39/KunQuant/runner/:${DYLD_LIBRARY_PATH}
fi
./build/temp.${SYSTEM_TRIPPLE}-cpython-39/KunCApiTest ./build/lib.${SYSTEM_TRIPPLE}-cpython-39/KunQuant/runner/libKunTest.${DYN_LIB_EXT} /tmp/alpha101_stream/alpha101_stream.${DYN_LIB_EXT}
echo "All test done"