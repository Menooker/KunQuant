#pragma once

#include "Base.hpp"

typedef void *KunExecutorHandle;
typedef void *KunLibraryHandle;
typedef void *KunModuleHandle;
typedef void *KunBufferNameMapHandle;
typedef void *KunStreamContextHandle;

#ifdef __cplusplus
extern "C" {
#endif

#define KUN_API_VERSION 1

#define KUN_INIT_NONE 0
#define KUN_INIT_FILE 1
#define KUN_INIT_MEMORY 2

typedef int KunStatus;

#define KUN_SUCCESS 0
#define KUN_INIT_ERROR 1
#define KUN_INVALID_ARGUMENT 2

typedef struct {
    size_t version;   // version of the KunQuant C API, must be set to
                      // KUN_API_VERSION
    size_t init_kind; // KUN_INIT_NONE KUN_INIT_FILE KUN_INIT_MEMORY
    union {
        const char *path; // path to stream state dump file
        struct {
            const char *buffer; // name of the stream state dump file
            size_t size;        // size of the stream state dump file
        } memory;               // memory buffer for stream state dump
    } init;
} KunStreamExtraArgs;

/**
 * @brief Create an single thread executor
 *
 * @return KunExecutorHandle It needs to be manually released by
 * kunDestoryExecutor
 */
KUN_API KunExecutorHandle kunCreateSingleThreadExecutor();

/**
 * @brief Create an multi-thread executor
 * @param numthreads the number of threads in the thread pool
 *
 * @return KunExecutorHandle It needs to be manually released by
 * kunDestoryExecutor
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
KUN_API KunLibraryHandle kunLoadLibrary(const char *path_or_name);

/**
 * @brief Find the module in the library by the name
 *
 * @param lib the loaded payload library
 * @param name the module name
 * @return KunModuleHandle. It shall not be free'd or released.
 */
KUN_API KunModuleHandle kunGetModuleFromLibrary(KunLibraryHandle lib,
                                                const char *name);

/**
 * @brief Unload the library
 *
 * @param ptr the loaded payload library
 */
KUN_API void kunUnloadLibrary(KunLibraryHandle ptr);

/**
 * @brief Create a map to store the mapping from the buffer name to the buffer
 * pointers
 *
 * @return KunBufferNameMapHandle. It needs to be released by
 * kunDestoryBufferNameMap
 */
KUN_API KunBufferNameMapHandle kunCreateBufferNameMap();

/**
 * @brief Relase the BufferNameMap
 */
KUN_API void kunDestoryBufferNameMap(KunBufferNameMapHandle ptr);

/**
 * @brief Set the buffer name to buffer mapping
 *
 * @param name the buffer name, which occurs in KunQuant code's Input(...) and
 * Output(...)
 * @param buffer the memory buffer. It must be large enough to hold the data.
 * Otherwise, memory overflow will occur.
 */
KUN_API void kunSetBufferNameMap(KunBufferNameMapHandle ptr, const char *name,
                                 float *buffer);

/**
 * @brief Delete a buffer name to buffer mapping
 *
 * @param name the buffer name, which occurs in KunQuant code's Input(...) and
 * Output(...)
 */
KUN_API void kunEraseBufferNameMap(KunBufferNameMapHandle ptr,
                                   const char *name);

/**
 * @brief Execute the computation graph. Regarding total_time, cur_time and
 * length: `total_time` is the dimension in "time" of the inputs. KunQuant will
 * compute from `cur_time` with `length` in the time dimension. And the outputs
 * will have `length` in time dimension. E.g. If you have input data of 500 rows
 * of time, and you want to compute factor starting from 100 to 250, you can set
 * `total_time=500, cur_time=100, length=150` If you want to compute the factors
 * for all of the time, set cur_time=0 and length=total_time
 *
 * @param exec the executor
 * @param m the module
 * @param buffers The inputs and outputs buffers
 * @param num_stocks The number of stocks. Must be multiple of 8
 * @param total_time See above brief
 * @param cur_time See above brief
 * @param length See above brief
 */
KUN_API void kunRunGraph(KunExecutorHandle exec, KunModuleHandle m,
                         KunBufferNameMapHandle buffers, size_t num_stocks,
                         size_t total_time, size_t cur_time, size_t length);
/**
 * @brief Create the Stream computing context.
 *
 * @param exec the executor
 * @param m the module
 * @param num_stocks The number of stocks.
 */
KUN_API KunStreamContextHandle kunCreateStream(KunExecutorHandle exec,
                                               KunModuleHandle m,
                                               size_t num_stocks);
/**
 * @brief Create the Stream computing context with extra arguments.
 *
 * @param exec the executor
 * @param m the module
 * @param num_stocks The number of stocks.
 * @param extra_args the extra arguments for stream context, to specify the
 * stream state dump file or memory buffer. The `version` field must be set to
 * KUN_API_VERSION. extra_args can be null for default behavior
 * @param out_handle the output handle to the stream context.
 * @return KUN_SUCCESS on success, KUN_INIT_ERROR if the stream context cannot
 * be initialized from the given states, KUN_INVALID_ARGUMENT if the extra_args
 * is invalid.
 */
KUN_API KunStatus kunCreateStreamEx(KunExecutorHandle exec, KunModuleHandle m,
                                    size_t num_stocks,
                                    const KunStreamExtraArgs *extra_args,
                                    KunStreamContextHandle *out_handle);

/**
 * @brief Serialize the states from the stream.
 *
 * @param context the stream object
 * @param dump_kind the kind of dump, KUN_INIT_FILE or KUN_INIT_MEMORY
 * @param path_or_buffer if dump_kind is KUN_INIT_FILE, this is the path to the
 * stream state dump file. If dump_kind is KUN_INIT_MEMORY, this is the memory
 * buffer to store the stream state dump. nullable if `*size` is 0 and dump_kind
 * is KUN_INIT_MEMORY.
 * @param size if dump_kind is KUN_INIT_MEMORY, this should point to the size of
 * the memory buffer. If the function succeeds or the buffer is too small, the
 * size will be set to the size of the stream state dump in bytes. If dump_kind
 * is KUN_INIT_FILE, this parameter is unused.
 * @return KUN_SUCCESS on success, KUN_INIT_ERROR if the memory buffer is too
 * small or there is a file error. If dump_kind is KUN_INIT_MEMORY `size` will
 * be overwritten to the size of dumped data. KUN_INVALID_ARGUMENT if dump_kind
 * is not KUN_INIT_FILE or KUN_INIT_MEMORY.
 */
KUN_API KunStatus kunStreamSerializeStates(KunStreamContextHandle context,
                                           size_t dump_kind,
                                           char *path_or_buffer, size_t *size);

/**
 * @brief Query the handle of a named buffer (input or output)
 * @param context the stream context
 * @param name the name of the input/output buffer, like "open", "alpha001"
 * @return the handle to the buffer, to be used in kunStreamGetCurrentBuffer and
 * kunStreamPushData
 * @note the complexity of the function is O(number_of_buffers). You should
 * cache the result of the handle to the named buffer to avoid frequently
 * calling this function. The result of this function for the same stream and
 * the same name will always remain unchanged.
 */
KUN_API size_t kunQueryBufferHandle(KunStreamContextHandle context,
                                    const char *name);

/**
 * @brief Get the memory address of a named buffer.
 * @param context the stream context
 * @param handle the named buffer handle. @see kunQueryBufferHandle
 * @return the pointer to the named buffer. The length the buffer
 * should be `num_stocks` elements of floats. Note that the pointer is valid
 * before calling kunStreamPushData or kunStreamRun
 */
KUN_API const float *kunStreamGetCurrentBuffer(KunStreamContextHandle context,
                                               size_t handle);

/**
 * @brief Copy the data to the named buffer of the stream. For each input named
 * buffer, you should call this function exactly once before calling
 * kunStreamRun
 * @param context the stream context
 * @param handle the named buffer handle. @see kunQueryBufferHandle
 * @param buffer the named buffer handle. The length the buffer should be
 * `num_stocks` elements of floats.
 */
KUN_API void kunStreamPushData(KunStreamContextHandle context, size_t handle,
                               const float *buffer);

/**
 * @brief Let the stream compute on the pushed data. All input data should be
 * updated via `kunStreamPushData` before calling this function. After this
 * function resturns, users can get the data via `kunStreamGetCurrentBuffer`
 * @param context the stream context
 */
KUN_API void kunStreamRun(KunStreamContextHandle context);

/**
 * @brief Release the stream
 * @param context the stream context
 */
KUN_API void kunDestoryStream(KunStreamContextHandle context);

#ifdef __cplusplus
}
#endif