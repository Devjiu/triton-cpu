import torch
import time
import triton


CPU_BLOCK_SIZE = 8192
CPU_TILE_SIZE = 16

torch.manual_seed(0)
# size = 98432
size = 65536

LINE_VALS = ["torch-cpu"]
LINE_NAMES = ["Torch Native"]
LINE_STYLES = [("blue", "-")]
# LINE_VALS = ['triton-cpu']
# LINE_NAMES = ['Triton CPU']
# LINE_STYLES = [('blue', '--')]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        # x_vals=[2**i for i in range(13, 28, 1)],  # Different possible values for `x_name`.
        x_vals=[
            2**i for i in [21]#range(16, 14, -1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f"vector-add-performance (CPU_BLOCK_SIZE={CPU_BLOCK_SIZE})",
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    import os

    device = "cpu" if "cpu" in provider else "cuda"
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)

    if device == "cpu":
        is_cpu = True
        triton.runtime.driver.set_active_to_cpu()
        if "single" in provider:
            os.environ["TRITON_CPU_SINGLE_CORE"] = "1"
        else:
            os.unsetenv("TRITON_CPU_SINGLE_CORE")
    else:
        is_cpu = False
        triton.runtime.driver.set_active_to_gpu()

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-cpu":
        # Note that we preallocate the output buffer here to only measure the kernel performance
        # without a large chunk of memory allocation.
        output = torch.empty_like(x)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.add(x, y, out=output), quantiles=quantiles, is_cpu=is_cpu
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
