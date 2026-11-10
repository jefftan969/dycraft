<!-- Mirrored from https://nvlabs.github.io/curobo/latest/reference/runtime_configuration.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# Runtime Configuration

cuRobo runtime configuration flags.

These module-level flags control runtime behavior. Set them in Python **before** creating any cuRobo objects:

    import curobo._src.runtime as runtime

    runtime.cuda_graphs = False   # disable CUDA graph capture
    runtime.torch_compile = True  # enable torch.compile

torch_compile: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable `torch.compile` for eligible cuRobo kernels.

torch_compile_slow: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable `torch.compile` for operations that are slow to compile.

torch_jit: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable `torch.jit.script` for eligible cuRobo functions.

cuda_graphs: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = True
Enable CUDA graph capture for solvers. Significantly improves throughput at the cost of a one-time warm-up overhead.

cuda_graph_reset: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable CUDA graph reset when tensor memory references change. Requires CUDA 12.0+.

cuda_streams: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = True
Enable separate CUDA streams for parallel kernel execution.

cuda_event_timers: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = True
Enable CUDA event-based timers for profiling.

cuda_core_backend: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = True
Enable the CUDA core runtime backend.

kernel_backend: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = 'auto'
Kernel backend to use. One of `"auto"`, `"cuda_core"`, or `"pybind"`. `"auto"` tries `cuda_core` first and falls back to `pybind`.

cache_dir: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = '/home/runner/.cache/curobo'
Directory for cuRobo outputs (example results, saved configs, etc.). Defaults to `~/.cache/curobo`.

debug: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable general debug mode (additional checks and logging).

debug_cuda_graphs: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable CUDA graph debug mode.

debug_cuda_compile: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable debug compilation flags for CUDA kernels.

debug_nan: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
If `True`, disable CUDA graphs and check for NaN values during optimization.

debug_timers: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable debug timing output and print results to stdout.

debug_trajopt: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Print trajectory optimization results for debugging.

profiler: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
Enable profiler hooks (e.g., for Nsight Systems).
