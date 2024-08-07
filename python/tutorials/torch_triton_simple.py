import torch

import triton
import triton.language as tl
import shutil
import time
import pandas as pd

import itt

class Event:
    def __init__(self):
        self.time = 0

    def elapsed_time(self, end_event) -> float:
        return (end_event.time - self.time) * 1000

    def record(self):
        self.time = time.perf_counter()

class ITTMock:
    def domain_create(self, *args, **kwargs):
        pass

    def task_begin(self, *args, **kwargs):
        pass

    def task_end(self, *args, **kwargs):
        pass

    def resume(self, *args, **kwargs):
        pass

    def pause(self, *args, **kwargs):
        pass


CPU_BLOCK_SIZE = 8192
CPU_TILE_SIZE = 16

itt_mock = ITTMock()

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               TILE_SIZE: tl.constexpr,  # Number of elements each loop iteration should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    for k in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + TILE_SIZE * k + tl.arange(0, TILE_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        # Write x + y back to DRAM.
        tl.store(output_ptr + offsets, output, mask=mask)

# @triton.jit
# def add_kernel_no_mask(x_ptr,  # *Pointer* to first input vector.
#                y_ptr,  # *Pointer* to second input vector.
#                output_ptr,  # *Pointer* to output vector.
#                n_elements,  # Size of the vector.
#                BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
#                TILE_SIZE: tl.constexpr,  # Number of elements each loop iteration should process.
#                # NOTE: `constexpr` so it can be used as a shape value.
#                ):
#     # There are multiple 'programs' processing different data. We identify which program
#     # we are here:
#     pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
#     # This program will process inputs that are offset from the initial data.
#     # For instance, if you had a vector of length 256 and block_size of 64, the programs
#     # would each access the elements [0:64, 64:128, 128:192, 192:256].
#     # Note that offsets is a list of pointers:
#     block_start = pid * BLOCK_SIZE
#     for k in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
#         offsets = block_start + TILE_SIZE * k + tl.arange(0, TILE_SIZE)
#         # Create a mask to guard memory operations against out-of-bounds accesses.
#         mask = offsets < n_elements
#         # Load x and y from DRAM, masking out any extra elements in case the input is not a
#         # multiple of the block size.
#         x = tl.load(x_ptr + offsets)
#         y = tl.load(y_ptr + offsets)
#         output = x + y
#         # Write x + y back to DRAM.
#         tl.store(output_ptr + offsets, output)

def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, is_cpu):
    # CPU_BLOCK_SIZE = 4096
    # CPU_TILE_SIZE = 16
    if output is None:
        # We need to preallocate the output.
        output = torch.empty_like(x)
        assert x.is_cpu == is_cpu and y.is_cpu == is_cpu and output.is_cpu == is_cpu
    n_elements = output.numel()
    block_size = 8192 # max(n_elements // 240, 8192)
    block_size = min(block_size, n_elements)
    # print("Block_size: ", block_size)
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    # grid = (triton.cdiv(n_elements, CPU_BLOCK_SIZE), )
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), )
    # print(grid)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    #assert n_elements % CPU_BLOCK_SIZE == 0
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size, TILE_SIZE=CPU_TILE_SIZE)
    #else:
    #add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=CPU_BLOCK_SIZE if is_cpu else GPU_BLOCK_SIZE, TILE_SIZE=CPU_TILE_SIZE)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


torch.manual_seed(0)
#size = 98432
# size = 65536

# # triton.runtime.driver.set_active_to_cpu()
# x = torch.rand(size, device='cpu')
# y = torch.rand(size, device='cpu')
# output_torch_cpu = torch.add(x, y)
# output_triton_cpu = add(x, y, None, is_cpu=True)
# print(output_torch_cpu)
# print(output_triton_cpu)
# print(f'The maximum difference between torch-cpu and triton-cpu is '
#       f'{torch.max(torch.abs(output_torch_cpu - output_triton_cpu))}')

LINE_VALS = ['triton-cpu']
LINE_NAMES = ['Triton CPU']
LINE_STYLES = [('blue', '--')]
# LINE_VALS = ['triton-cpu', 'torch-cpu']
# LINE_NAMES = ['Triton CPU', 'Torch Native']
# LINE_STYLES = [('blue', '--'), ('blue', '-')]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        #x_vals=[2**i for i in range(13, 28, 1)],  # Different possible values for `x_name`.
        x_vals=[2**i for i in range(16, 14, -1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'vector-add-performance (CPU_BLOCK_SIZE={CPU_BLOCK_SIZE})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    import os

    device = 'cpu' if 'cpu' in provider else 'cuda'
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)

    if device == 'cpu':
        is_cpu = True
        triton.runtime.driver.set_active_to_cpu()
        if 'single' in provider:
            os.environ['TRITON_CPU_SINGLE_CORE'] = '1'
        else:
            os.unsetenv('TRITON_CPU_SINGLE_CORE')
    else:
        is_cpu = False
        triton.runtime.driver.set_active_to_gpu()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton-cpu':
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y, output, True), quantiles=quantiles,
                                                     is_cpu=is_cpu)
    elif provider == 'torch-cpu':
        # Note that we preallocate the output buffer here to only measure the kernel performance
        # without a large chunk of memory allocation.
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.add(x, y, out=output), quantiles=quantiles,
                                                     is_cpu=is_cpu)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
# benchmark.run(print_data=True, show_plots=True)

# @triton.jit
# def kernel_fill_zero(out, BLOCK_SIZE: tl.constexpr):
#     start = tl.program_id(0)
#     offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     zero = tl.full((BLOCK_SIZE, ), 0, dtype=tl.int32)
#     tl.store(out + offs, zero)

# cache_size = 512 * 1024 * 1024
# cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")
# def zero_cache():
#     assert cache.shape[0] % 1024 == 0
#     kernel_fill_zero[(cache.shape[0] // 1024, )](cache, 1024)
#     cache.zero_()

cache_domain = itt_mock.domain_create("cache") 
add_domain = itt_mock.domain_create("add")

# triton.runtime.driver.set_active_to_cpu()

def test_add(size):
    #n_rows = 65536
    print(f"Running Triton CPU with {size} ...")
    device = "cpu"
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)
    output = torch.empty_like(x)

    n_warmup = 10
    n_repeat = 500
    start_event = [Event() for i in range(n_repeat)]
    end_event = [Event() for i in range(n_repeat)]

    cache_size = 512 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")
    
    for _ in range(n_warmup):
        add(x, y, output, True)

    itt.resume()
    for i in range(n_repeat):
        itt_mock.task_begin(cache_domain, "reset cache")
        cache.zero_()
        itt_mock.task_end(cache_domain)

        start_event[i].record()
        itt_mock.task_begin(add_domain, "run")
        add(x, y, output, True)
        itt_mock.task_end(add_domain)
        end_event[i].record()
    itt.pause()

    quantiles = [0.5, 0.2, 0.8]
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    ms, min_ms, max_ms = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    #print(f"{gbps(ms)}, {gbps(max_ms)}, {gbps(min_ms)}")
    return gbps(ms), gbps(max_ms), gbps(min_ms), ms, max_ms, min_ms

res = pd.DataFrame(columns={'Size':int, 'Med (GBps)':float, 'Min (GBps)':float, 'Max (GBps)':float, 'Med (Ms)':float, 'Min (Ms)':float, 'Max (Ms)':float})
# for i in [1024]: # [128 * i for i in range(2, 34, 1)]:
    # med, min, max = test_add(i)
    # res.loc[len(res.index)] = [i, med, min, max]

size = 2**21
med, min, max, med_ms, min_ms, max_ms = test_add(size)
res.loc[len(res.index)] = [size, med, min, max, med_ms, min_ms, max_ms]

print(res)
