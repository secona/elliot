set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Elliot)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Elliot)
add_public_tablegen_target(ElliotPassIncGen)
