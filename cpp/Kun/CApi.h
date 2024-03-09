#pragma once

#include "Base.hpp"

typedef void* KunExecutorHandle;
typedef void* KunLibraryHandle;
typedef void* KunModuleHandle;
typedef void* KunBufferNameMapHandle;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create an single thread executor
 * 
 * @return KunExecutorHandle It needs to be manually released by kunDestoryExecutor
 */
KUN_API KunExecutorHandle kunCreateSingleThreadExecutor();

/**
 * @brief Create an multi-thread executor
 * @param numthreads the number of threads in the thread pool
 * 
 * @return KunExecutorHandle It needs to be manually released by kunDestoryExecutor
 */
KUN_API KunExecutorHandle kunCreateMultiThreadExecutor(int numthreads);

/**
 * @brief Release the executor
 * 
 * @param ptr 
 * @return KUN_API 
 */
KUN_API void kunDestoryExecutor(KunExecutorHandle ptr);

/**
 * @brief Load the payload library compiled by KunQuant
 * 
 * @param path_or_name The path to the shared library file
 * @return KunLibraryHandle It needs to be released via kunUnloadLibrary
 */
KUN_API KunLibraryHandle kunLoadLibrary(const char* path_or_name);

/**
 * @brief Find the module in the library by the name
 * 
 * @param lib the loaded payload library
 * @param name the module name
 * @return KunModuleHandle. It shall not be free'd or released. 
 */
KUN_API KunModuleHandle kunGetModuleFromLibrary(KunLibraryHandle lib, const char* name);

/**
 * @brief Unload the library
 * 
 * @param ptr the loaded payload library
 */
KUN_API void kunUnloadLibrary(KunLibraryHandle ptr);

/**
 * @brief Create a map to store the mapping from the buffer name to the buffer pointers
 * 
 * @return KunBufferNameMapHandle. It needs to be released by kunDestoryBufferNameMap
 */
KUN_API KunBufferNameMapHandle kunCreateBufferNameMap();

/**
 * @brief Relase the BufferNameMap
 */
KUN_API void kunDestoryBufferNameMap(KunBufferNameMapHandle ptr);

/**
 * @brief Set the buffer name to buffer mapping
 * 
 * @param name the buffer name, which occurs in KunQuant code's Input(...) and Output(...)
 * @param buffer the memory buffer. It must be large enough to hold the data. Otherwise, memory overflow will occur.
 */
KUN_API void kunSetBufferNameMap(KunBufferNameMapHandle ptr, const char* name, float* buffer);

/**
 * @brief Delete a buffer name to buffer mapping
 * 
 * @param name the buffer name, which occurs in KunQuant code's Input(...) and Output(...)
 */
KUN_API void kunEraseBufferNameMap(KunBufferNameMapHandle ptr, const char* name);

/**
 * @brief Execute the computation graph. Regarding total_time, cur_time and length:
 * `total_time` is the dimension in "time" of the inputs. KunQuant will compute from
 * `cur_time` with `length` in the time dimension. And the outputs will have `length`
 * in time dimension. E.g. If you have input data of 500 rows of time, and you want to
 * compute factor starting from 100 to 250, you can set `total_time=500, cur_time=100, length=150`
 * If you want to compute the factors for all of the time, set cur_time=0 and length=total_time
 * 
 * @param exec the executor
 * @param m the module
 * @param buffers The inputs and outputs buffers
 * @param num_stocks The number of stocks
 * @param total_time See above brief
 * @param cur_time See above brief
 * @param length See above brief
 */
KUN_API void kunRunGraph(KunExecutorHandle exec, KunModuleHandle m,
              KunBufferNameMapHandle buffers,
              size_t num_stocks, size_t total_time, size_t cur_time,
              size_t length);

#ifdef __cplusplus
}
#endif