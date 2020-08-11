set(LIBS

  MLIRCIM
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransformUtils
)
  
add_llvm_library(MLIRCIMUtils
  RuntimeUtils.cpp
  StaticUtils.cpp
)

add_dependencies(MLIRCIMUtils ${LIBS})
target_link_libraries(MLIRCIMUtils ${LIBS})
