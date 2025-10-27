"""
Async Engine (без ccxt.pro): паралелимо CPU-bound/Python I/O через ThreadPool.
Мета — не блокувати головну петлю, знизити латентність на багато символів.
"""
import concurrent.futures as cf
import os

MAX_WORKERS = int(os.getenv("ASYNC_MAX_WORKERS", "8"))

_executor = cf.ThreadPoolExecutor(max_workers=MAX_WORKERS)

def run_async(func, *args, **kwargs):
    """Запускає функцію у пулі, повертає майбутній об’єкт."""
    return _executor.submit(func, *args, **kwargs)

def map_async(func, iterable):
    """Паралельне виконання списку задач."""
    futures = [run_async(func, it) for it in iterable]
    return futures
