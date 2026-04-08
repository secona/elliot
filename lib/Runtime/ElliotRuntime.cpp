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

// The raw CUDA kernel string.
// Later, you will inject '#define' macros here before compilation to optimize
// it.
const char *spmv_kernel_code = R"CUDA(
extern "C" __global__ void spmv_csr(
    int num_rows, 
    const int64_t* ptr, 
    const int64_t* idx, 
    const float* val, 
    const float* x, 
    float* y) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int64_t start = ptr[row];
        int64_t end = ptr[row + 1];
        
        for (int64_t i = start; i < end; ++i) {
            sum += val[i] * x[idx[i]];
        }
        y[row] = sum;
    }
}
)CUDA";

extern "C" void
elliot_jit_spmv_csr_f32(MemRef1D<int64_t> *ptr_ref, MemRef1D<int64_t> *idx_ref,
                        MemRef1D<float> *val_ref, MemRef1D<float> *x_ref,
                        MemRef1D<float> *y_ref, int64_t num_rows) {

  std::cout << "[Elliot] Intercepted SpMV. Triggering NVRTC JIT..."
            << std::endl;

  // 1. Initialize CUDA Driver API
  CUDA_SAFE_CALL(cuInit(0));
  CUdevice cuDevice;
  CUcontext context;
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

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

  // 6. Set up Kernel Arguments and Launch
  // Extract raw device pointers from MLIR memrefs
  int64_t *d_ptr = ptr_ref->alignedPtr + ptr_ref->offset;
  int64_t *d_idx = idx_ref->alignedPtr + idx_ref->offset;
  float *d_val = val_ref->alignedPtr + val_ref->offset;
  float *d_x = x_ref->alignedPtr + x_ref->offset;
  float *d_y = y_ref->alignedPtr + y_ref->offset;
  int rows_int = static_cast<int>(num_rows);

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

  // Clean up context
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
}
