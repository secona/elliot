#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mlir/Dialect/SparseTensor/IR/Enums.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/SparseTensor/File.h>
#include <mlir/ExecutionEngine/SparseTensorRuntime.h>
#include <string>

// 1D MemRef for the dense vectors
typedef StridedMemRefType<float, 1> MemRef1D;

extern "C" {
  // Our JIT-intercepted MLIR function
  void _mlir_ciface_test_spmv(MemRef1D *out, void *a, MemRef1D *x, MemRef1D *y_in);
}

void *create_csr_tensor(std::string filename) {
  // Hardcoded to 8x8 to match test.mlir
  uint64_t sizes_data[] = {8, 8};
  StridedMemRefType<uint64_t, 1> sizes_memref = {
      sizes_data, sizes_data, 0, {2}, {1}};

  LevelType lvl_types_data[] = {LevelType(LevelFormat::Dense),
                                LevelType(LevelFormat::Compressed)};
  StridedMemRefType<LevelType, 1> lvl_types_memref = {
      lvl_types_data, lvl_types_data, 0, {2}, {1}};

  uint64_t mapping_data[] = {0, 1};
  StridedMemRefType<uint64_t, 1> mapping_memref = {
      mapping_data, mapping_data, 0, {2}, {1}};

  uint64_t dimRank = 2;
  uint64_t dimShape[2] = {8, 8};
  PrimaryType valTp = PrimaryType::kF32;

  SparseTensorReader *reader =
      SparseTensorReader::create(filename.c_str(), dimRank, dimShape, valTp);

  void *tensor = _mlir_ciface_newSparseTensor(
      &sizes_memref, &sizes_memref, &lvl_types_memref, &mapping_memref,
      &mapping_memref, mlir::sparse_tensor::OverheadType::kIndex,
      mlir::sparse_tensor::OverheadType::kIndex, valTp,
      mlir::sparse_tensor::Action::kFromReader, reader);

  uint64_t nse = getSparseTensorReaderNSE(reader);
  std::cout << "[Host] non-zeros read: " << nse << std::endl;

  delSparseTensorReader(reader);
  return tensor;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./elliot_app <matrix.mtx>" << std::endl;
    return 1;
  }

  void *sparse_tensor_a = create_csr_tensor(argv[1]);
  assert(sparse_tensor_a);

  // Allocate dense inputs on the heap so MLIR can manage them cleanly
  float *x_data = (float *)malloc(8 * sizeof(float));
  float *y_in_data = (float *)malloc(8 * sizeof(float));
  for(int i = 0; i < 8; ++i) {
    x_data[i] = 1.0f;    // Input vector
    y_in_data[i] = 0.0f; // Accumulator
  }

  MemRef1D x_ref = {x_data, x_data, 0, {8}, {1}};
  MemRef1D y_in_ref = {y_in_data, y_in_data, 0, {8}, {1}};

  // Allocate the output struct pointer
  MemRef1D *out = (MemRef1D *)malloc(sizeof(MemRef1D));

  std::cout << "[Host] Passing control to MLIR & Elliot JIT..." << std::endl;

  // Execute the pipeline
  _mlir_ciface_test_spmv(out, sparse_tensor_a, &x_ref, &y_in_ref);

  std::cout << "[Host] Execution complete. Results:\n";
  std::cout << "sizes: " << out->sizes[0] << "\n";
  std::cout << "strides: " << out->strides[0] << "\n";

  float *data = out->data;
  for (int i = 0; i < out->sizes[0]; i++) {
    int idx = out->offset + i * out->strides[0];
    std::cout << "y[" << i << "] = " << data[idx] << "\n";
  }

  // Cleanup
  std::free(out->basePtr);
  std::free(out);
  std::free(x_data);
  std::free(y_in_data);

  // Use whatever delete function your headers mapped this to
  // (usually _mlir_ciface_delSparseTensor in newer commits)
  // delSparseTensor(sparse_tensor_a);
}
