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
def add_kernel_no_mask(x_ptr,  # *Pointer* to first input vector.
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
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        output = x + y
        # Write x + y back to DRAM.
        tl.store(output_ptr + offsets, output)

def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, is_cpu):
    if output is None:
        # We need to preallocate the output.
        output = torch.empty_like(x)
        assert x.is_cpu == is_cpu and y.is_cpu == is_cpu and output.is_cpu == is_cpu
    n_elements = output.numel()
    block_size = 8192 # max(n_elements // 120, 8192)
    block_size = min(block_size, n_elements)
    # print("block size: ", block_size, " size: ", x.size())
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    # grid = (triton.cdiv(n_elements, CPU_BLOCK_SIZE), )
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']), )
    add_kernel_no_mask[grid](x, y, output, n_elements, BLOCK_SIZE=block_size, TILE_SIZE=CPU_TILE_SIZE)
    return output


torch.manual_seed(0)

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
