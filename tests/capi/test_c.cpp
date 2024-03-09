#include <Kun/CApi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CHECK(V) if(!(V)) {printf("CHECK(" #V ") faild\n"); return 3;} 

int main(int args, char **argv) {
    if (args != 2) {
        printf("Bad args\n");
        return 2;
    }

    // prepare inputs
    const size_t num_stocks = 24;
    const size_t num_time = 10;
    // dimension: [3 x 10 x 8]
    float *inputs = new float[num_stocks/8*10*8];
    for(size_t i=0;i<num_stocks*num_time; i++) {
        inputs[i] = float(rand()) / RAND_MAX;
    }
    float *outputs = new float[num_stocks/8*10*8];

    KunExecutorHandle exec = kunCreateSingleThreadExecutor();
    CHECK(exec);
    KunLibraryHandle lib = kunLoadLibrary(argv[1]);
    CHECK(lib);
    KunModuleHandle modu = kunGetModuleFromLibrary(lib, "testRuntimeModule");
    CHECK(modu);
    KunBufferNameMapHandle bufs = kunCreateBufferNameMap();
    CHECK(bufs);

    kunSetBufferNameMap(bufs, "input", inputs);
    kunSetBufferNameMap(bufs, "output", outputs);

    kunRunGraph(exec, modu, bufs, num_stocks, num_time, 0, num_time);

    for(size_t i=0;i<num_stocks*num_time; i++) {
        if (std::abs(outputs[i] - inputs[i] * 3) > 1e-5) {
            printf("Output error at %zu => %f, %f\n", i, outputs[i], inputs[i]);
            return 4;
        }
    }

    delete []inputs;
    delete []outputs;
    kunDestoryBufferNameMap(bufs);
    kunUnloadLibrary(lib);
    kunDestoryExecutor(exec);
    printf("Test done\n");
    return 0;
}