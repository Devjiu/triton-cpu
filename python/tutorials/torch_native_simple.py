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
        torch.add(x, y, out=output)

    itt_mock.resume()
    for i in range(n_repeat):
        itt_mock.task_begin(cache_domain, "reset cache")
        # zero_cache() # 
        cache.zero_()
        itt_mock.task_end(cache_domain)0

        start_event[i].record()
        itt_mock.task_begin(add_domain, "run")
        torch.add(x, y, out=output)
        itt_mock.task_end(add_domain)
        end_event[i].record()
    itt_mock.pause()

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
