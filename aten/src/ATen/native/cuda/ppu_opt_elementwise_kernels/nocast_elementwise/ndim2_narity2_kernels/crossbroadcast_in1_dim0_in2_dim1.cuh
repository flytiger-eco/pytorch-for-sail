
// Cross broadcast: arg0 dim0-broadcast (scalar read per row) and arg1
// dim1-broadcast (vector read on dim0, reused across the row loop).
template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    func_t f, int64_t y_t, int64_t y_remain) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nt + threadIdx.x) * vt;
  if (x >= size0) {
    return;
  }
  int64_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }
  const int64_t y_start = static_cast<int64_t>(blockIdx.y) * y_t;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  const vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + x * stride02);
  #pragma unroll 2
  for (int64_t y_idx = 0; y_idx < y_loop; y_idx++) {
    const int64_t y = y_start + y_idx;
    const arg0_t in1 = c10::load(reinterpret_cast<const arg0_t*>(data1 + y * stride11));
    vec_res out;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      out.val[i] = f(in1, in2.val[i]);
    }
    *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
  }
}

// float64 小方阵的 x/y 共同映射。原路径的 size0=128、vt=2 仅有 64 个
// x vectors，且每条线程串行处理 8 行；该路径将 4 行交给不同 lane，仍由
// 每个 x lane 复用其 in2 vector，并完整保留传入的 PyTorch functor。
template <int nx, int ny, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
C10_LAUNCH_BOUNDS_2(nx * ny, 4)
__global__ void elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1_narrow(
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride02, int64_t stride10, int64_t stride11,
    func_t f) {
  const int64_t x = (static_cast<int64_t>(blockIdx.x) * nx + threadIdx.x) * vt;
  const int64_t y = static_cast<int64_t>(blockIdx.y) * ny + threadIdx.y;
  if (x >= size0 || y >= size1) return;
  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  const arg0_t in1 = c10::load(reinterpret_cast<const arg0_t*>(data1 + y * stride11));
  const vec_arg1 in2 = *reinterpret_cast<const vec_arg1*>(data2 + x * stride02);
  vec_res out;
#pragma unroll
  for (int i = 0; i < vt; ++i) out.val[i] = f(in1, in2.val[i]);
  *reinterpret_cast<vec_res*>(data0 + y * stride10 + x * stride00) = out;
}


template <int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static void launch_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1(
    int64_t N,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  if constexpr (sizeof(res_t) == 8 && vt == 2) {
    if (size0 <= 256) {
      constexpr int nx = 32, ny = 4;
      dim3 block(nx, ny);
      dim3 grid((size0 / vt + nx - 1) / nx, (size1 + ny - 1) / ny);
      auto stream = at::cuda::getCurrentCUDAStream();
      elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1_narrow<nx, ny, vt, res_t, arg0_t, arg1_t, func_t>
          <<<grid, block, 0, stream>>>(data0, data1, data2, size0, size1,
              stride00, stride02, stride10, stride11, f);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  constexpr int64_t y_t = 8;
  const int64_t grid_x =
      (size0 + static_cast<int64_t>(nt) * vt - 1) / (static_cast<int64_t>(nt) * vt);
  const int64_t grid_y = (size1 + y_t - 1) / y_t;
  const int64_t y_remain = size1 - (grid_y - 1) * y_t;
  dim3 block(nt);
  dim3 grid(grid_x, grid_y);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, vt, res_t, arg0_t, arg1_t, func_t>
      <<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f, y_t, y_remain);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
