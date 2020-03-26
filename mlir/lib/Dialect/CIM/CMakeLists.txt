add_llvm_library(MLIRCIM
  IR/CIMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CIM

  DEPENDS
  MLIRCIMOpsIncGen
  )
target_link_libraries(MLIRCIM
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )
