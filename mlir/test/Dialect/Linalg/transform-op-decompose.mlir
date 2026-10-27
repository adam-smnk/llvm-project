// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// Expected indexing maps for batchless conv_1d_nwc_wcf.
// CHECK-DAG:  #[[$CONV_I:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d3)>
// CHECK-DAG:  #[[$CONV_F:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK-DAG:  #[[$CONV_O:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// Expected indexing maps for batchless depthwise_conv_1d_wc_wcf.
// CHECK-DAG:  #[[$DW_I:.+]] = affine_map<(d0, d1, d2) -> (d0 + d2, d1)>
// CHECK-DAG:  #[[$DW_F:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>

// Expected indexing maps for batchless pooling_cw_min.
// CHECK-DAG:  #[[$POOL_I:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 + d2)>
// CHECK-DAG:  #[[$POOL_F:.+]] = affine_map<(d0, d1, d2) -> (d2)>

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// Expected indexing maps for 1D conv (cross-conv after downscale from generic).
// CHECK-DAG:  #[[$CROSS_1D_I:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:  #[[$CROSS_1D_F:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[$CROSS_1D_O:.+]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @conv_2d_nhwc_hwcf
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?x?x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @conv_2d_nhwc_hwcf(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?x?x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d_nwc_wcf
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?x?x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @conv_2d_nchw_fchw
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @conv_2d_nchw_fchw(%input: tensor<?x?x1x?xf32>, %filter: tensor<?x?x1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d_ncw_fcw
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<?x?x1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

// Depthwise conv with height=1 (downscales height dimension)
// CHECK-LABEL: @depthwise_conv_2d_nhwc_hwc_height
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x1x113x96xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x3x96xf32>
func.func @depthwise_conv_2d_nhwc_hwc_height(%input: tensor<1x1x113x96xf32>, %filter: tensor<1x3x96xf32>) -> tensor<1x1x56x96xf32> {
  // CHECK: %[[RES:.+]] = tensor.empty
  %init = tensor.empty() : tensor<1x1x56x96xf32>
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICERES:.+]] = tensor.extract_slice %[[RES]]
  // CHECK: %[[OPRES:.+]] = linalg.depthwise_conv_1d_nwc_wc
  // CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]]
  // CHECK-SAME: outs(%[[SLICERES]]
  // CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[OPRES]] into %[[RES]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x1x113x96xf32>, tensor<1x3x96xf32>)
         outs(%init: tensor<1x1x56x96xf32>) -> tensor<1x1x56x96xf32>
  // CHECK: %[[INSERTED]]
  return %0: tensor<1x1x56x96xf32>
}

// Depthwise conv with width=1 (downscales width dimension)
// CHECK-LABEL: @depthwise_conv_2d_nhwc_hwc_width
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x113x1x96xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<3x1x96xf32>
func.func @depthwise_conv_2d_nhwc_hwc_width(%input: tensor<1x113x1x96xf32>, %filter: tensor<3x1x96xf32>) -> tensor<1x56x1x96xf32> {
  // CHECK: %[[RES:.+]] = tensor.empty
  %init = tensor.empty() : tensor<1x56x1x96xf32>
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICERES:.+]] = tensor.extract_slice %[[RES]]
  // CHECK: %[[OPRES:.+]] = linalg.depthwise_conv_1d_nwc_wc
  // CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]]
  // CHECK-SAME: outs(%[[SLICERES]]
  // CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[OPRES]] into %[[RES]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x113x1x96xf32>, tensor<3x1x96xf32>)
         outs(%init: tensor<1x56x1x96xf32>) -> tensor<1x56x1x96xf32>
  // CHECK: %[[INSERTED]]
  return %0: tensor<1x56x1x96xf32>
}

// CHECK-LABEL: @conv_2d
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<1x?xf32>)
func.func @conv_2d(%input: tensor<1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<1x?xf32>) -> tensor<1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d
     ins (%input, %filter: tensor<1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<1x?xf32>) -> tensor<1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<1x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_sum
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_sum(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_sum
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nchw_sum
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @pooling_nchw_sum(%input: tensor<?x?x1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_ncw_sum
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_max
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_max(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_max
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_max_unsigned
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xi32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xi32>
func.func @pooling_nhwc_max_unsigned(%input: tensor<?x1x?x?xi32>, %filter: tensor<1x?xi32>, %init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_max_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xi32>, tensor<1x?xi32>)
    outs (%init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xi32>
}

// CHECK-LABEL: @pooling_nhwc_min
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_min(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_min
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_min_unsigned
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xi32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xi32>
func.func @pooling_nhwc_min_unsigned(%input: tensor<?x1x?x?xi32>, %filter: tensor<1x?xi32>, %init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_min_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xi32>, tensor<1x?xi32>)
    outs (%init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xi32>
}

// CHECK-LABEL: @pooling_nchw_max
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @pooling_nchw_max(%input: tensor<?x?x1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_ncw_max
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

#map_conv_i = affine_map<(oh, ow, f, kh, kw, c) -> (oh + kh, ow + kw, c)>
#map_conv_f = affine_map<(oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>
#map_conv_o = affine_map<(oh, ow, f, kh, kw, c) -> (oh, ow, f)>

// CHECK-LABEL: @batchless_conv_2d_hwc_hwcf
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x14x8xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3x8x16xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12x16xf32>
func.func @batchless_conv_2d_hwc_hwcf(%input: tensor<1x14x8xf32>, %filter: tensor<1x3x8x16xf32>, %output: tensor<1x12x16xf32>) -> tensor<1x12x16xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$CONV_I]], #[[$CONV_F]], #[[$CONV_O]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_conv_i, #map_conv_f, #map_conv_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x14x8xf32>, tensor<1x3x8x16xf32>)
    outs(%output : tensor<1x12x16xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12x16xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<1x12x16xf32>
}

#map_dw_i = affine_map<(oh, ow, c, kh, kw) -> (oh + kh, ow + kw, c)>
#map_dw_f = affine_map<(oh, ow, c, kh, kw) -> (kh, kw, c)>
#map_dw_o = affine_map<(oh, ow, c, kh, kw) -> (oh, ow, c)>

// CHECK-LABEL: @batchless_depthwise_conv_2d_hwc_hwc
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x14x8xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3x8xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12x8xf32>
func.func @batchless_depthwise_conv_2d_hwc_hwc(%input: tensor<1x14x8xf32>, %filter: tensor<1x3x8xf32>, %output: tensor<1x12x8xf32>) -> tensor<1x12x8xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$DW_I]], #[[$DW_F]], #[[$MAP1]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_dw_i, #map_dw_f, #map_dw_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x14x8xf32>, tensor<1x3x8xf32>)
    outs(%output : tensor<1x12x8xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12x8xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<1x12x8xf32>
}

#map_pool_i = affine_map<(c, oh, ow, kh, kw) -> (c, oh + kh, ow + kw)>
#map_pool_f = affine_map<(c, oh, ow, kh, kw) -> (kh, kw)>
#map_pool_o = affine_map<(c, oh, ow, kh, kw) -> (c, oh, ow)>

// CHECK-LABEL: @batchless_pooling_chw_min
// CHECK-SAME:    %[[ARG0:.+]]: tensor<8x1x14xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<8x1x12xf32>
func.func @batchless_pooling_chw_min(%input: tensor<8x1x14xf32>, %filter: tensor<1x3xf32>, %output: tensor<8x1x12xf32>) -> tensor<8x1x12xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$POOL_I]], #[[$POOL_F]], #[[$MAP1]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_pool_i, #map_pool_f, #map_pool_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<8x1x14xf32>, tensor<1x3xf32>)
    outs(%output : tensor<8x1x12xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %min = arith.minimumf %out, %in : f32
    linalg.yield %min : f32
  } -> tensor<8x1x12xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<8x1x12xf32>
}

#map_cross_i = affine_map<(d0, d1, d2, d3) -> (d0 + d3, d1 + d2)>
#map_cross_f = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map_cross_o = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: @cross_conv_nonstandard_loop_order
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x15xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x1xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12xf32>
func.func @cross_conv_nonstandard_loop_order(%input: tensor<1x15xf32>, %filter: tensor<3x1xf32>, %output: tensor<1x12xf32>) -> tensor<1x12xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$CROSS_1D_I]], #[[$CROSS_1D_F]], #[[$CROSS_1D_O]]]
  // CHECK-SAME:    iterator_types = ["parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  // CHECK:       return %[[RES]]
  %0 = linalg.generic {
    indexing_maps = [#map_cross_i, #map_cross_f, #map_cross_o],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x15xf32>, tensor<3x1xf32>)
    outs(%output : tensor<1x12xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12xf32>
  return %0 : tensor<1x12xf32>
}

func.func @softmax(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL:      func.func @softmax(
// CHECK-SAME:           %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>, %[[DST:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-DAG:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D4:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[D3]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          linalg.yield %[[D9]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D6:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[D4]] : tensor<2x16x32xf32>) outs(%[[D5]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[D4]], %[[D6]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.divf %[[IN]], %[[IN_1]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        return %[[D7]] : tensor<2x16x32xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose %0 : (!transform.any_op) -> !transform.any_op

    %2 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.decompose_interface %2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Dimension scaling of parallel dims - scales are applied to contraction
// results accumulated exactly in i32.

func.func @scaled_contract_dim_scaling(
    %A: tensor<8x16xi8>, %sA: tensor<8xf8E8M0FNU>,
    %B: tensor<4x16xi8>, %sB: tensor<4xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x16xi8>, tensor<8xf8E8M0FNU>, tensor<4x16xi8>, tensor<4xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// CHECK-DAG: #[[$CONTRACT_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$CONTRACT_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$CONTRACT_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$SCALE_OUT:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$SCALE_A:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$SCALE_B:.+]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func @scaled_contract_dim_scaling
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<8x16xi8>
//  CHECK-SAME:   %[[SA:[a-zA-Z0-9]+]]: tensor<8xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: tensor<4x16xi8>
//  CHECK-SAME:   %[[SB:[a-zA-Z0-9]+]]: tensor<4xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: tensor<8x4xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x4xi32>
//       CHECK:   %[[ZERO:.+]] = arith.constant 0 : i32
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[EMPTY]] : tensor<8x4xi32>) -> tensor<8x4xi32>
//       CHECK:   %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:     indexing_maps = [#[[$CONTRACT_A]], #[[$CONTRACT_B]], #[[$CONTRACT_C]]]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<8x16xi8>, tensor<4x16xi8>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<8x4xi32>)
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$SCALE_OUT]], #[[$SCALE_A]], #[[$SCALE_B]], #[[$SCALE_OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[CONTRACT]], %[[SA]], %[[SB]] : tensor<8x4xi32>, tensor<8xf8E8M0FNU>, tensor<4xf8E8M0FNU>)
//  CHECK-SAME:     outs(%[[C]] : tensor<8x4xf32>)
//       CHECK:   ^bb0(%[[IN:.+]]: i32, %[[IN_SA:.+]]: f8E8M0FNU, %[[IN_SB:.+]]: f8E8M0FNU, %[[OUT:.+]]: f32):
//       CHECK:     %[[IN_FP:.+]] = arith.sitofp %[[IN]] : i32 to f32
//       CHECK:     %[[SA_EXT:.+]] = arith.extf %[[IN_SA]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SA:.+]] = arith.mulf %[[IN_FP]], %[[SA_EXT]] : f32
//       CHECK:     %[[SB_EXT:.+]] = arith.extf %[[IN_SB]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_EXT]] : f32
//       CHECK:     %[[ACC:.+]] = arith.addf %[[OUT]], %[[MUL_SB]] : f32
//       CHECK:     linalg.yield %[[ACC]] : f32
//       CHECK:   return %[[SCALED]] : tensor<8x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// MXFP4 block scaling of the reduction dim - the reduction dim is split into
// blocks; the contraction reduces within blocks and the scaled block results
// are reduced by the epilogue.

func.func @scaled_contract_mxfp4(
    %A: tensor<8x64xf4E2M1FN>, %sA: tensor<8x2xf8E8M0FNU>,
    %B: tensor<4x64xf4E2M1FN>, %sB: tensor<4x2xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 32)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x64xf4E2M1FN>, tensor<8x2xf8E8M0FNU>, tensor<4x64xf4E2M1FN>, tensor<4x2xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// CHECK-DAG: #[[$CONTRACT_A:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$CONTRACT_B:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[$CONTRACT_T:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$SCALE_T:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$SCALE_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$SCALE_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$SCALE_OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @scaled_contract_mxfp4
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<8x64xf4E2M1FN>
//  CHECK-SAME:   %[[SA:[a-zA-Z0-9]+]]: tensor<8x2xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: tensor<4x64xf4E2M1FN>
//  CHECK-SAME:   %[[SB:[a-zA-Z0-9]+]]: tensor<4x2xf8E8M0FNU>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: tensor<8x4xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x4x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<8x4x2xf32>)
//       CHECK:   %[[EXP_A:.+]] = tensor.expand_shape %[[A]] {{\[}}[0], [1, 2]] output_shape [8, 2, 32]
//       CHECK:   %[[EXP_B:.+]] = tensor.expand_shape %[[B]] {{\[}}[0], [1, 2]] output_shape [4, 2, 32]
//       CHECK:   %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:     indexing_maps = [#[[$CONTRACT_A]], #[[$CONTRACT_B]], #[[$CONTRACT_T]]]
//  CHECK-SAME:     ins(%[[EXP_A]], %[[EXP_B]] : tensor<8x2x32xf4E2M1FN>, tensor<4x2x32xf4E2M1FN>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<8x4x2xf32>)
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$SCALE_T]], #[[$SCALE_A]], #[[$SCALE_B]], #[[$SCALE_OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[CONTRACT]], %[[SA]], %[[SB]] : tensor<8x4x2xf32>, tensor<8x2xf8E8M0FNU>, tensor<4x2xf8E8M0FNU>)
//  CHECK-SAME:     outs(%[[C]] : tensor<8x4xf32>)
//       CHECK:   ^bb0(%[[IN:.+]]: f32, %[[IN_SA:.+]]: f8E8M0FNU, %[[IN_SB:.+]]: f8E8M0FNU, %[[OUT:.+]]: f32):
//       CHECK:     %[[SA_EXT:.+]] = arith.extf %[[IN_SA]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SA:.+]] = arith.mulf %[[IN]], %[[SA_EXT]] : f32
//       CHECK:     %[[SB_EXT:.+]] = arith.extf %[[IN_SB]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_EXT]] : f32
//       CHECK:     %[[ACC:.+]] = arith.addf %[[OUT]], %[[MUL_SB]] : f32
//       CHECK:     linalg.yield %[[ACC]] : f32
//       CHECK:   return %[[SCALED]] : tensor<8x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Nested block sizes on the reduction dim and block scaling of a parallel dim.
// The dims are split into multiple block levels, scales and output are
// reshaped accordingly. Mixed integer and floating-point inputs accumulate in
// f32. Non-f8E8M0FNU scale of floating-point data is rounded to f8E8M0FNU
// like in `arith.scaling_extf`.

func.func @scaled_contract_nested_blocks(
    %A: tensor<8x128xi8>, %sA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<4x128xf8E4M3FN>, %sB: tensor<2x2xf16>,
    %C: tensor<8x4xf16>) -> tensor<8x4xf16> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n floordiv 2, k floordiv 64)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x128xi8>, tensor<8x4xf8E8M0FNU>, tensor<4x128xf8E4M3FN>, tensor<2x2xf16>)
      outs(%C : tensor<8x4xf16>) -> tensor<8x4xf16>
  return %D : tensor<8x4xf16>
}

// CHECK-DAG: #[[$CONTRACT_A:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
// CHECK-DAG: #[[$CONTRACT_B:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[$CONTRACT_T:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[$SCALE_T:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[$SCALE_A:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK-DAG: #[[$SCALE_B:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3)>
// CHECK-DAG: #[[$SCALE_OUT:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK-LABEL: func @scaled_contract_nested_blocks
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<8x128xi8>
//  CHECK-SAME:   %[[SA:[a-zA-Z0-9]+]]: tensor<8x4xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: tensor<4x128xf8E4M3FN>
//  CHECK-SAME:   %[[SB:[a-zA-Z0-9]+]]: tensor<2x2xf16>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: tensor<8x4xf16>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x2x2x2x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<8x2x2x2x2xf32>)
//       CHECK:   %[[EXP_A:.+]] = tensor.expand_shape %[[A]] {{\[}}[0], [1, 2, 3]] output_shape [8, 2, 2, 32]
//       CHECK:   %[[EXP_B:.+]] = tensor.expand_shape %[[B]] {{\[}}[0, 1], [2, 3, 4]] output_shape [2, 2, 2, 2, 32]
//       CHECK:   %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:     indexing_maps = [#[[$CONTRACT_A]], #[[$CONTRACT_B]], #[[$CONTRACT_T]]]
//  CHECK-SAME:     ins(%[[EXP_A]], %[[EXP_B]] : tensor<8x2x2x32xi8>, tensor<2x2x2x2x32xf8E4M3FN>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<8x2x2x2x2xf32>)
//       CHECK:   %[[EXP_C:.+]] = tensor.expand_shape %[[C]] {{\[}}[0], [1, 2]] output_shape [8, 2, 2]
//       CHECK:   %[[EXP_SA:.+]] = tensor.expand_shape %[[SA]] {{\[}}[0], [1, 2]] output_shape [8, 2, 2]
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$SCALE_T]], #[[$SCALE_A]], #[[$SCALE_B]], #[[$SCALE_OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:     ins(%[[CONTRACT]], %[[EXP_SA]], %[[SB]] : tensor<8x2x2x2x2xf32>, tensor<8x2x2xf8E8M0FNU>, tensor<2x2xf16>)
//  CHECK-SAME:     outs(%[[EXP_C]] : tensor<8x2x2xf16>)
//       CHECK:   ^bb0(%[[IN:.+]]: f32, %[[IN_SA:.+]]: f8E8M0FNU, %[[IN_SB:.+]]: f16, %[[OUT:.+]]: f16):
//       CHECK:     %[[SA_EXT:.+]] = arith.extf %[[IN_SA]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SA:.+]] = arith.mulf %[[IN]], %[[SA_EXT]] : f32
//       CHECK:     %[[SB_E8M0:.+]] = arith.truncf %[[IN_SB]] : f16 to f8E8M0FNU
//       CHECK:     %[[SB_EXT:.+]] = arith.extf %[[SB_E8M0]] : f8E8M0FNU to f32
//       CHECK:     %[[MUL_SB:.+]] = arith.mulf %[[MUL_SA]], %[[SB_EXT]] : f32
//       CHECK:     %[[OUT_EXT:.+]] = arith.extf %[[OUT]] : f16 to f32
//       CHECK:     %[[ACC:.+]] = arith.addf %[[OUT_EXT]], %[[MUL_SB]] : f32
//       CHECK:     %[[RES:.+]] = arith.truncf %[[ACC]] : f32 to f16
//       CHECK:     linalg.yield %[[RES]] : f16
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SCALED]] {{\[}}[0], [1, 2]] : tensor<8x2x2xf16> into tensor<8x4xf16>
//       CHECK:   return %[[COLLAPSED]] : tensor<8x4xf16>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Block scale spanning the whole reduction dim, e.g., after tiling, is
// broadcast. Tensor scaling with a scalar scale and transposed output.

func.func @scaled_contract_uniform_block(
    %A: tensor<8x32xf8E5M2>, %sA: tensor<8x1xf8E8M0FNU>,
    %B: tensor<32x4xf8E5M2>, %sB: f8E8M0FNU,
    %C: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> ()>,
                       affine_map<(m, n, k) -> (n, m)>]
      ins(%A, %sA, %B, %sB : tensor<8x32xf8E5M2>, tensor<8x1xf8E8M0FNU>, tensor<32x4xf8E5M2>, f8E8M0FNU)
      outs(%C : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %D : tensor<4x8xf32>
}

// CHECK-DAG: #[[$CONTRACT_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$CONTRACT_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$CONTRACT_T:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[$SCALE_OUT:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$SCALE_A:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG: #[[$SCALE_B:.+]] = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func @scaled_contract_uniform_block
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<8x32xf8E5M2>
//  CHECK-SAME:   %[[SA:[a-zA-Z0-9]+]]: tensor<8x1xf8E8M0FNU>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]+]]: tensor<32x4xf8E5M2>
//  CHECK-SAME:   %[[SB:[a-zA-Z0-9]+]]: f8E8M0FNU
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]+]]: tensor<4x8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x8xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<4x8xf32>)
//       CHECK:   %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:     indexing_maps = [#[[$CONTRACT_A]], #[[$CONTRACT_B]], #[[$CONTRACT_T]]]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<8x32xf8E5M2>, tensor<32x4xf8E5M2>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<4x8xf32>)
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$SCALE_OUT]], #[[$SCALE_A]], #[[$SCALE_B]], #[[$SCALE_OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[CONTRACT]], %[[SA]], %[[SB]] : tensor<4x8xf32>, tensor<8x1xf8E8M0FNU>, f8E8M0FNU)
//  CHECK-SAME:     outs(%[[C]] : tensor<4x8xf32>)
//       CHECK:   return %[[SCALED]] : tensor<4x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Block scaling of a dynamic parallel dim is applied by the epilogue as is.
// Accumulation and scaling of a wide output are performed in output precision.

func.func @scaled_contract_dynamic_parallel_block(
    %A: tensor<?x64xi8>, %sA: tensor<?x2xf8E8M0FNU>,
    %B: tensor<4x64xi8>, %sB: tensor<4x2xf8E8M0FNU>,
    %C: tensor<?x4xf64>) -> tensor<?x4xf64> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m floordiv 16, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 32)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<?x64xi8>, tensor<?x2xf8E8M0FNU>, tensor<4x64xi8>, tensor<4x2xf8E8M0FNU>)
      outs(%C : tensor<?x4xf64>) -> tensor<?x4xf64>
  return %D : tensor<?x4xf64>
}

// CHECK-DAG: #[[$SCALE_A:.+]] = affine_map<(d0, d1, d2) -> (d0 floordiv 16, d2)>

// CHECK-LABEL: func @scaled_contract_dynamic_parallel_block
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: tensor<?x64xi8>
//  CHECK-SAME:   %[[SA:[a-zA-Z0-9]+]]: tensor<?x2xf8E8M0FNU>
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[A]], %{{.+}} : tensor<?x64xi8>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?x4x2xi32>
//       CHECK:   %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:     ins({{.*}} : tensor<?x2x32xi8>, tensor<4x2x32xi8>)
//  CHECK-SAME:     -> tensor<?x4x2xi32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#{{.+}}, #[[$SCALE_A]], #{{.+}}, #{{.+}}]
//  CHECK-SAME:     ins(%[[CONTRACT]], %[[SA]], %{{.+}} : tensor<?x4x2xi32>, tensor<?x2xf8E8M0FNU>, tensor<4x2xf8E8M0FNU>)
//       CHECK:     arith.sitofp %{{.+}} : i32 to f64
//       CHECK:     arith.extf %{{.+}} : f8E8M0FNU to f64

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Tiling of the block scaled reduction dim with tile size being a multiple of
// the block size followed by decomposition of the tiled op.

func.func @scaled_contract_tiled(
    %A: tensor<16x256xf8E4M3FN>, %sA: tensor<16x8xf8E8M0FNU>,
    %B: tensor<16x256xf8E4M3FN>, %sB: tensor<16x8xf8E8M0FNU>,
    %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 32)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<16x256xf8E4M3FN>, tensor<16x8xf8E8M0FNU>, tensor<16x256xf8E4M3FN>, tensor<16x8xf8E8M0FNU>)
      outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %D : tensor<16x16xf32>
}

// CHECK-LABEL: func @scaled_contract_tiled
//       CHECK:   scf.for {{.*}} iter_args(%[[ACC:.+]] = %{{.+}}) -> (tensor<16x16xf32>)
//       CHECK:     %[[SLICE_A:.+]] = tensor.extract_slice {{.*}} to tensor<16x64xf8E4M3FN>
//       CHECK:     %[[SLICE_SA:.+]] = tensor.extract_slice {{.*}} to tensor<16x2xf8E8M0FNU>
//       CHECK:     %[[SLICE_B:.+]] = tensor.extract_slice {{.*}} to tensor<16x64xf8E4M3FN>
//       CHECK:     %[[SLICE_SB:.+]] = tensor.extract_slice {{.*}} to tensor<16x2xf8E8M0FNU>
//       CHECK:     %[[SLICE_C:.+]] = tensor.extract_slice %[[ACC]]
//       CHECK:     %[[EXP_A:.+]] = tensor.expand_shape %[[SLICE_A]] {{\[}}[0], [1, 2]] output_shape [16, 2, 32]
//       CHECK:     %[[EXP_B:.+]] = tensor.expand_shape %[[SLICE_B]] {{\[}}[0], [1, 2]] output_shape [16, 2, 32]
//       CHECK:     %[[CONTRACT:.+]] = linalg.contract
//  CHECK-SAME:       ins(%[[EXP_A]], %[[EXP_B]] : tensor<16x2x32xf8E4M3FN>, tensor<16x2x32xf8E4M3FN>)
//  CHECK-SAME:       -> tensor<16x16x2xf32>
//       CHECK:     %[[SCALED:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       ins(%[[CONTRACT]], %[[SLICE_SA]], %[[SLICE_SB]] :
//  CHECK-SAME:       outs(%[[SLICE_C]] : tensor<16x16xf32>)
//       CHECK:     tensor.insert_slice %[[SCALED]] into %[[ACC]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %0 tile_sizes [0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1 = transform.structured.decompose_interface %tiled : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Unsupported decompositions fail gracefully without modifying the op.

// Block scaling of a dynamic reduction dim.
// CHECK-LABEL: func @scaled_contract_dynamic_reduction_block
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_dynamic_reduction_block(
    %A: tensor<8x?xf4E2M1FN>, %sA: tensor<8x?xf8E8M0FNU>,
    %B: tensor<4x?xf4E2M1FN>, %sB: tensor<4x?xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 32)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x?xf4E2M1FN>, tensor<8x?xf8E8M0FNU>, tensor<4x?xf4E2M1FN>, tensor<4x?xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// Reduction dim size not divisible by the block size.
// CHECK-LABEL: func @scaled_contract_partial_block
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_partial_block(
    %A: tensor<8x100xf4E2M1FN>, %sA: tensor<8x4xf8E8M0FNU>,
    %B: tensor<4x100xf4E2M1FN>, %sB: tensor<4x4xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 32)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x100xf4E2M1FN>, tensor<8x4xf8E8M0FNU>, tensor<4x100xf4E2M1FN>, tensor<4x4xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// Block sizes of the reduction dim which do not divide each other.
// CHECK-LABEL: func @scaled_contract_incompatible_blocks
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_incompatible_blocks(
    %A: tensor<8x96xf4E2M1FN>, %sA: tensor<8x3xf8E8M0FNU>,
    %B: tensor<4x96xf4E2M1FN>, %sB: tensor<4x2xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m, k floordiv 32)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n, k floordiv 48)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x96xf4E2M1FN>, tensor<8x3xf8E8M0FNU>, tensor<4x96xf4E2M1FN>, tensor<4x2xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// Scaling of every element along the reduction dim leaves nothing to contract.
// CHECK-LABEL: func @scaled_contract_reduction_dim_scaling
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_reduction_dim_scaling(
    %A: tensor<8x16xi8>, %sA: tensor<16xf8E8M0FNU>,
    %B: tensor<4x16xi8>, %sB: tensor<4xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x16xi8>, tensor<16xf8E8M0FNU>, tensor<4x16xi8>, tensor<4xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// Scale of floating-point data which cannot be rounded to f8E8M0FNU.
// CHECK-LABEL: func @scaled_contract_unsupported_scale_type
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_unsupported_scale_type(
    %A: tensor<8x16xf8E4M3FN>, %sA: tensor<8xf8E5M2>,
    %B: tensor<4x16xf8E4M3FN>, %sB: tensor<4xf8E8M0FNU>,
    %C: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %D = linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : tensor<8x16xf8E4M3FN>, tensor<8xf8E5M2>, tensor<4x16xf8E4M3FN>, tensor<4xf8E8M0FNU>)
      outs(%C : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %D : tensor<8x4xf32>
}

// Memref semantics.
// CHECK-LABEL: func @scaled_contract_memref
//  CHECK-NEXT:   linalg.scaled_contract
func.func @scaled_contract_memref(
    %A: memref<8x16xi8>, %sA: memref<8xf8E8M0FNU>,
    %B: memref<4x16xi8>, %sB: memref<4xf8E8M0FNU>,
    %C: memref<8x4xf32>) {
  linalg.scaled_contract
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (m)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %sA, %B, %sB : memref<8x16xi8>, memref<8xf8E8M0FNU>, memref<4x16xi8>, memref<4xf8E8M0FNU>)
      outs(%C : memref<8x4xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.sequence %arg1 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      %0 = transform.structured.match ops{["linalg.scaled_contract"]} in %root : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.decompose_interface %0 : (!transform.any_op) -> !transform.any_op
    }
    transform.yield
  }
}
