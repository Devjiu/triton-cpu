#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <ittnotify.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <vector>

extern "C" void add_kernel_no_mask(float *x_ptr, float *y_ptr, float *out_ptr,
                                   int32_t n_elements, uint32_t x, uint32_t y,
                                   uint32_t z, uint32_t grid_x, uint32_t grid_y,
                                   uint32_t grid_z);

using kernel_ptr_t = void (*)(float *x_ptr, float *y_ptr, float *out_ptr,
                              int32_t n_elements, uint32_t x, uint32_t y,
                              uint32_t z, uint32_t grid_x, uint32_t grid_y,
                              uint32_t grid_z);

extern "C" void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                kernel_ptr_t kernel_ptr, float *x_ptr,
                                float *y_ptr, float *out_ptr, int n_elements);

void run_add_kernel_triton_launcher(float *in0, float *in1, float *out,
                                    int32_t n_elements, int N) {
  run_omp_kernels(N, 1, 1, add_kernel_no_mask, in0, in1, out, n_elements);
}

void run_add_kernels(float *in0, float *in1, float *out, int32_t n_elements,
                     int N) {
  // __itt_task_begin(dom1, __itt_null, __itt_null, omp_launcher_call_str);
  //#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    // __itt_task_begin(dom1, __itt_null, __itt_null, kernel_call_str);
    add_kernel_no_mask(in0, in1, out, n_elements, i, 0, 0, N, 0, 0);
    // __itt_task_end(dom1);
  }
  // __itt_task_end(dom1);
}

using DurationMs = std::chrono::duration<double, std::milli>;

constexpr size_t SIZE = std::pow(2, 21); // 4096 * 4096;
constexpr size_t ITERS = 500;
constexpr size_t WMUP = 10;

double toGbs(DurationMs t, size_t num_bytes) {
  // milisec = 10^(-3) sec , GBs = 10^9 bytes
  return double(num_bytes) / (t.count() / 1000) / (1000 * 1000 * 1000);
}

double toGibs(DurationMs t, size_t num_bytes) {
  return double(num_bytes) / (t.count() / 1000) / (1024 * 1024 * 1024);
}

DurationMs getQuantileVal(long max_index, float quantile,
                          std::vector<DurationMs> &sorted_t) {
  float possible_ind = max_index * quantile;
  if (possible_ind - (long)possible_ind == 0.0) {
    return sorted_t[(long)possible_ind];
  }
  long lower_ind = (long)possible_ind;
  long upper_ind = (long)possible_ind + 1;
  return sorted_t[lower_ind] +
         (sorted_t[upper_ind] - sorted_t[lower_ind]) * quantile;
}

std::string toPerfString(std::vector<DurationMs> &t, size_t num_bytes) {
  if (t.size() == 0) {
    return "nope";
  }

  std::sort(t.begin(), t.end());
  DurationMs min = t.front();
  DurationMs max = t.back();
  int mid = t.size() / 2;
  DurationMs med = t.size() % 2 == 0 ? (t[mid] + t[mid - 1]) / 2 : t[mid];
  float min_quantile = 0.2;
  float max_quantile = 0.8;

  DurationMs min_quant = getQuantileVal(t.size() - 1, min_quantile, t);
  DurationMs max_quant = getQuantileVal(t.size() - 1, max_quantile, t);

  std::stringstream ss;
  ss << "        Size  Med (GBps)  Min (GBps)  Max (GBps)  Med (Ms)  Min (Ms)  "
        "Max (Ms)\n";
  ss << "         " << t.size() << "     " << toGbs(med, num_bytes) << "     "
     << toGbs(min_quant, num_bytes) << "     " << toGbs(max_quant, num_bytes)
     << "   " << med.count() << "   " << min_quant.count() << "   "
     << max_quant.count() << "\n";
  ss << "Med: " << med.count() << "ms MinQ: " << max_quant.count()
     << "ms MaxQ: " << min_quant.count() << "ms Min: " << max.count()
     << "ms Max: " << min.count() << "ms " << std::endl;
  ss << "Med: " << toGbs(med, num_bytes)
     << "GBps MinQ: " << toGbs(min_quant, num_bytes)
     << "GBps MaxQ: " << toGbs(max_quant, num_bytes)
     << "GBps Min: " << toGbs(max, num_bytes)
     << "GBps Max: " << toGbs(min, num_bytes) << "GBps";
  return ss.str();
}

void __attribute__((noinline)) clear_cache_fast() {
  auto start = std::chrono::high_resolution_clock::now();
  constexpr size_t CACHE_SIZE = (512 << 20) / (sizeof(int)); // 512MB
  static std::unique_ptr<int[]> cache_data(new int[CACHE_SIZE]);
  //#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < CACHE_SIZE; ++i) {
    cache_data[i] = 0;
  }
  auto end = std::chrono::high_resolution_clock::now();
  DurationMs diff = end - start;
  // std::cout << "Clear cache took: " << diff.count() << "ms " << std::endl;
}

#define CHECK_CORRECTNESS 0
#define PROFILING 0
#define TRITON_LAUNCHER 1
#define CLEAR_CACHE 1

void measure(float *in0, float *in1, float *out, int32_t n_elements, int N) {
  std::vector<DurationMs> res(ITERS);
  std::cout << "iters: " << ITERS << " v size: " << res.size() << "\n";
  for (size_t i = 0; i < WMUP; ++i) {
    run_add_kernels(in0, in1, out, n_elements, N);
  }

  __itt_resume();
  for (size_t i = 0; i < ITERS; ++i) {
#if CLEAR_CACHE
    clear_cache_fast();
#endif
    auto start = std::chrono::high_resolution_clock::now();
    //__itt_task_begin(dom1, __itt_null, __itt_null, launcher_call_str);
    // run_add_kernels(in0, in1, out, n_elements, N);
    run_add_kernels(in0, in1, out, n_elements, N);
    //__itt_task_end(dom1);
    auto end = std::chrono::high_resolution_clock::now();
    res[i] = end - start;
    // std::cout << "i: " << i << " val: " << res[i].count() << std::endl;
  }
  __itt_pause();

  size_t num_bytes = 3 * n_elements * sizeof(float);
  std::cout << "Input size - " << n_elements << " N - " << N << std::endl;
  std::cout << toPerfString(res, num_bytes) << std::endl;
}

int main(int argc, const char **argv) {
  std::vector<float> in0(SIZE + 15, 1.00010);
  std::vector<float> in1(SIZE + 15, 2.00002);
  std::vector<float> out(SIZE + 15, -1);
  std::vector<float> expected_out(SIZE + 15, 3.00012);

  float *in0_ptr = reinterpret_cast<float *>(
      reinterpret_cast<size_t>(in0.data() + 15) & 0xffffffffffffffc0);
  float *in1_ptr = reinterpret_cast<float *>(
      reinterpret_cast<size_t>(in1.data() + 15) & 0xffffffffffffffc0);
  float *out_ptr = reinterpret_cast<float *>(
      reinterpret_cast<size_t>(out.data() + 15) & 0xffffffffffffffc0);

  for (int i = 0; i < SIZE; ++i) {
    in0_ptr[i] = in0_ptr[i] + i * 0.2;
    in1_ptr[i] = in1_ptr[i] + i * 0.2;
    // out_ptr[i] = -2;
    expected_out[i] = in0_ptr[i] + in1_ptr[i];
  }
  int32_t block_size = 8192;
  uint32_t N = std::ceil(SIZE / float(block_size));

#if CHECK_CORRECTNESS
  std::cout << "size " << SIZE << " N " << N << std::endl;
  std::cout << "Checking correctness... \n";
  measure(in0_ptr, in1_ptr, out_ptr, SIZE, N);
  // add_ref(in0_ptr, in1_ptr, out_ptr, 2048, 3712);
  for (int block_idx = 0; block_idx < N; ++block_idx) {
    float ref = out_ptr[block_idx * block_size];
    float exp = expected_out[block_idx * block_size];
    if (abs(ref - exp) > 0.00001) {
      std::cout << "Error at (" << block_idx << "): expected - " << exp
                << " != taken - " << ref << " abs = " << abs(ref - exp)
                << std::endl;
      abort();
    }
  }
  std::cout << "PASS" << std::endl;
#endif

  measure(in0_ptr, in1_ptr, out_ptr, SIZE, N);
}
