from pynvml import *

def print_gpu_utilization(verbose=True):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    if verbose:
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used//1024**2


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
