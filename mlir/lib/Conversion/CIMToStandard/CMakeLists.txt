add_llvm_library(MLIRCIMToStandardTransforms
  CIMToStandardPass.cpp
  )

target_link_libraries(MLIRCIMToStandardTransforms
  LLVMSupport
  MLIRCIM
  MLIRStandardOps
  MLIRPass
  MLIRTransformUtils
  )
