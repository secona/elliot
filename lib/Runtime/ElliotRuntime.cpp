#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <vector>

// Standard MLIR 1D Memref Descriptor
template <typename T> struct MemRef1D {
  T *allocatedPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

// Error checking macros (essential for NVRTC debugging)
#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "NVRTC error: " << nvrtcGetErrorString(result)              \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorString(result, &msg);                                          \
      std::cerr << "CUDA error: " << msg << std::endl;                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const char *spmv_kernel_code = R"CUDA(
extern "C" __global__ void spmv_csr(
    int num_rows, 
    const long long* ptr, 
    const long long* idx, 
    const float* val, 
    const float* x, 
    float* y) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        long long start = ptr[row];
        long long end = ptr[row + 1];

        for (long long i = start; i < end; ++i) {
            sum += val[i] * x[idx[i]];
        }

        y[row] = sum;
    }
}
)CUDA";

extern "C" void
_mlir_ciface_elliot_jit_spmv_csr_f32(MemRef1D<int64_t> *ptr_ref, MemRef1D<int64_t> *idx_ref,
                        MemRef1D<float> *val_ref, MemRef1D<float> *x_ref,
                        MemRef1D<float> *y_ref, int64_t num_rows) {

  std::cout << "[Elliot] Intercepted SpMV. Triggering NVRTC JIT..."
            << std::endl;

  // 1. Initialize CUDA Driver API
  CUDA_SAFE_CALL(cuInit(0));
  CUdevice cuDevice;
  CUcontext context;
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

#if CUDA_VERSION >= 13000
  CUctxCreateParams ctxParams{};
  CUDA_SAFE_CALL(cuCtxCreate(&context, &ctxParams, 0, cuDevice));
#else
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
#endif

  // 2. Create the NVRTC Program
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, spmv_kernel_code, "spmv_csr.cu", 0,
                                     NULL, NULL));

  // 3. Compile the Program
  // Target the Turing architecture (sm_75) for your 2080 Ti
  const char *opts[] = {"--gpu-architecture=compute_75"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(prog, log.data());
    std::cerr << "Compile Log: \n" << log.data() << std::endl;
    exit(1);
  }

  // 4. Extract PTX Assembly
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  // 5. Load PTX into CUDA Module
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "spmv_csr"));

  // Extract raw HOST pointers from MLIR memrefs
  int64_t *h_ptr = ptr_ref->alignedPtr + ptr_ref->offset;
  int64_t *h_idx = idx_ref->alignedPtr + idx_ref->offset;
  float *h_val = val_ref->alignedPtr + val_ref->offset;
  float *h_x = x_ref->alignedPtr + x_ref->offset;
  float *h_y = y_ref->alignedPtr + y_ref->offset;
  int rows_int = static_cast<int>(num_rows);

  // Calculate byte sizes based on the memref sizes array
  size_t ptr_bytes = ptr_ref->sizes[0] * sizeof(int64_t);
  size_t idx_bytes = idx_ref->sizes[0] * sizeof(int64_t);
  size_t val_bytes = val_ref->sizes[0] * sizeof(float);
  size_t x_bytes = x_ref->sizes[0] * sizeof(float);
  size_t y_bytes = y_ref->sizes[0] * sizeof(float);

  // Allocate DEVICE memory
  CUdeviceptr d_ptr, d_idx, d_val, d_x, d_y;
  CUDA_SAFE_CALL(cuMemAlloc(&d_ptr, ptr_bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_idx, idx_bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_val, val_bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_x, x_bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_y, y_bytes));

  // Copy Host to Device
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_ptr, h_ptr, ptr_bytes));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_idx, h_idx, idx_bytes));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_val, h_val, val_bytes));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_x, h_x, x_bytes));

  // Set up Kernel Arguments (using device pointers)
  void *args[] = {&rows_int, &d_ptr, &d_idx, &d_val, &d_x, &d_y};

  int threadsPerBlock = 256;
  int blocksPerGrid = (rows_int + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "[Elliot] Launching Kernel. Blocks: " << blocksPerGrid
            << std::endl;

  CUDA_SAFE_CALL(cuLaunchKernel(kernel, blocksPerGrid, 1, 1, // grid dim
                                threadsPerBlock, 1, 1,       // block dim
                                0, NULL,   // shared mem and stream
                                args, 0)); // arguments

  CUDA_SAFE_CALL(cuCtxSynchronize());

  CUDA_SAFE_CALL(cuMemcpyDtoH(h_y, d_y, y_bytes));

  CUDA_SAFE_CALL(cuMemFree(d_ptr));
  CUDA_SAFE_CALL(cuMemFree(d_idx));
  CUDA_SAFE_CALL(cuMemFree(d_val));
  CUDA_SAFE_CALL(cuMemFree(d_x));
  CUDA_SAFE_CALL(cuMemFree(d_y));
  
  // Clean up context
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
}
