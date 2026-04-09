../build/tools/elliot-opt/elliot-opt --convert-spmm-to-elliot kernel.mlir > program.mlir
../third_party/llvm-project/build/bin/mlir-opt --sparsifier program.mlir > program1.mlir
../third_party/llvm-project/build/bin/mlir-translate --mlir-to-llvmir program1.mlir > program.ll
../third_party/llvm-project/build/bin/llc program.ll -O3 -filetype=obj -o program.o

clang++ main.cpp program.o \
  -I../third_party/llvm-project/mlir/include \
  -L../build/lib \
  -L../third_party/llvm-project/build/lib \
  -lElliotRuntime \
  -lmlir_c_runner_utils \
  -lMLIRSparseTensorRuntime \
  -Wl,-rpath,../build/lib \
  -Wl,-rpath,../third_party/llvm-project/build/lib \
  -o elliot_app
