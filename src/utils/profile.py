import time
import psutil
import os
import threading
from functools import wraps


def profile(func):
    """Decorator to profile time, peak RAM, and average CPU usage of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        peak_memory = 0
        cpu_usage_samples = []
        running = True

        def monitor():
            nonlocal peak_memory
            while running:
                mem = process.memory_info().rss  # in bytes
                peak_memory = max(peak_memory, mem)

                cpu = process.cpu_percent(interval=0.1)
                cpu_usage_samples.append(cpu)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            running = False
            monitor_thread.join()

        total_time = end_time - start_time
        peak_memory_mb = peak_memory / (1024**2)
        avg_cpu = (
            sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0
        )

        print(f"--------Profiling results for {func.__name__}:---------")
        print(f"Time taken: {total_time:.4f} sec")
        print(f"Peak RAM: {peak_memory_mb:.2f} MB")
        print(f"Avg CPU usage: {avg_cpu:.2f}%")

        return result

    return wrapper
