set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XeGPU)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXeGPUPassIncGen)
add_dependencies(mlir-headers MLIRXeGPUPassIncGen)

add_mlir_doc(Passes XeGPUPasses ./ -gen-pass-doc)
