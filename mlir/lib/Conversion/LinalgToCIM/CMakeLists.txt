add_llvm_library(MLIRLinalgToCIMTransforms
  LinalgToCIMPass.cpp
  )

target_link_libraries(MLIRLinalgToCIMTransforms
  LLVMSupport
  MLIRLinalgOps
  MLIRCIM
  MLIRPass
  MLIRTransformUtils
  )
