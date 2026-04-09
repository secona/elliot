#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

func.func @test_spmv(
  %A: tensor<8x8xf32, #CSR>,
  %x: tensor<8xf32>,
  %y_in: tensor<8xf32>
) -> tensor<8xf32> attributes {llvm.emit_c_interface} {
  %y_out = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%A, %x : tensor<8x8xf32, #CSR>, tensor<8xf32>)
    outs(%y_in : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %c, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>

  return %y_out : tensor<8xf32>
}
