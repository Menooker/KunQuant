#include <Kun/CApi.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(V)                                                               \
    if (!(V)) {                                                                \
        printf("CHECK(" #V ") faild\n");                                       \
        return 3;                                                              \
    }

static int testBatch(const char *libpath) {
    // prepare inputs
    const size_t num_stocks = 24;
    const size_t num_time = 10;
    // dimension: [3 x 10 x 8]
    float *inputs = new float[num_stocks / 8 * 10 * 8];
    for (size_t i = 0; i < num_stocks * num_time; i++) {
        inputs[i] = float(rand()) / RAND_MAX;
    }
    float *outputs = new float[num_stocks / 8 * 10 * 8];

    KunExecutorHandle exec = kunCreateSingleThreadExecutor();
    CHECK(exec);
    KunLibraryHandle lib = kunLoadLibrary(libpath);
    CHECK(lib);
    KunModuleHandle modu = kunGetModuleFromLibrary(lib, "testRuntimeModule");
    CHECK(modu);
    KunBufferNameMapHandle bufs = kunCreateBufferNameMap();
    CHECK(bufs);

    kunSetBufferNameMap(bufs, "input", inputs);
    kunSetBufferNameMap(bufs, "output", outputs);

    kunRunGraph(exec, modu, bufs, num_stocks, num_time, 0, num_time);

    for (size_t i = 0; i < num_stocks * num_time; i++) {
        if (std::abs(outputs[i] - inputs[i] * 3) > 1e-5) {
            printf("Output error at %zu => %f, %f\n", i, outputs[i], inputs[i]);
            return 4;
        }
    }

    delete[] inputs;
    delete[] outputs;
    kunDestoryBufferNameMap(bufs);
    kunUnloadLibrary(lib);
    kunDestoryExecutor(exec);
    printf("Test done: batch\n");
    return 0;
}

// check alpha101 = (self.close - self.open) /((self.high - self.low) + 0.001)
static int testStream(const char *libpath) {
    // prepare inputs
    const size_t num_stocks = 24;
    float *dataclose = new float[num_stocks];
    float *dataopen = new float[num_stocks];
    float *datahigh = new float[num_stocks];
    float *datalow = new float[num_stocks];
    float *datavol = new float[num_stocks];
    float *dataamount = new float[num_stocks];
    for (size_t i = 0; i < num_stocks; i++) {
        dataclose[i] = float(rand()) / RAND_MAX;
        dataopen[i] = float(rand()) / RAND_MAX;
        datahigh[i] = float(rand()) / RAND_MAX;
        datalow[i] = float(rand()) / RAND_MAX;
        datavol[i] = float(rand()) / RAND_MAX;
        dataamount[i] = float(rand()) / RAND_MAX;
    }
    float *alpha101 = new float[num_stocks];

    KunExecutorHandle exec = kunCreateSingleThreadExecutor();
    CHECK(exec);
    KunLibraryHandle lib = kunLoadLibrary(libpath);
    CHECK(lib);
    KunModuleHandle modu = kunGetModuleFromLibrary(lib, "alpha_101_stream");
    CHECK(modu);
    KunStreamContextHandle ctx = kunCreateStream(exec, modu, num_stocks);
    CHECK(ctx);

    size_t handleClose = kunQueryBufferHandle(ctx, "close");
    size_t handleOpen = kunQueryBufferHandle(ctx, "open");
    size_t handleHigh = kunQueryBufferHandle(ctx, "high");
    size_t handleLow = kunQueryBufferHandle(ctx, "low");
    size_t handleVol = kunQueryBufferHandle(ctx, "volume");
    size_t handleAmount = kunQueryBufferHandle(ctx, "amount");
    size_t handleAlpha101 = kunQueryBufferHandle(ctx, "alpha101");
    // don't need to query the handles everytime when calling kunStreamPushData

    kunStreamPushData(ctx, handleClose, dataclose);
    kunStreamPushData(ctx, handleOpen, dataopen);
    kunStreamPushData(ctx, handleHigh, datahigh);
    kunStreamPushData(ctx, handleLow, datalow);
    kunStreamPushData(ctx, handleVol, datavol);
    kunStreamPushData(ctx, handleAmount, dataamount);

    kunStreamRun(ctx);
    memcpy(alpha101, kunStreamGetCurrentBuffer(ctx, handleAlpha101),
           sizeof(float) * num_stocks);

    for (size_t i = 0; i < num_stocks; i++) {
        float expected =
            (dataclose[i] - dataopen[i]) / (datahigh[i] - datalow[i] + 0.001);
        if (std::abs(alpha101[i] - expected) > 1e-5) {
            printf("Output error at %zu => %f, %f\n", i, alpha101[i], expected);
            return 4;
        }
    }

    delete[] dataclose;
    delete[] dataopen;
    delete[] datahigh;
    delete[] datalow;
    delete[] datavol;
    delete[] dataamount;
    kunDestoryStream(ctx);
    kunUnloadLibrary(lib);
    kunDestoryExecutor(exec);
    printf("Test done: stream\n");
    return 0;
}

int main(int args, char **argv) {
    if (args != 3) {
        printf("Bad args\n");
        return 2;
    }
    auto ret = testBatch(argv[1]);
    if (ret) {
        return ret;
    }
    ret = testStream(argv[2]);
    if (ret) {
        return ret;
    }

    return 0;
}