import time
from asyncio.coroutines import iscoroutinefunction


def timeit(func):
    def run(*args, **kwargs):
        s = time.perf_counter()
        result = func(*args, **kwargs)
        e = time.perf_counter()
        print(rf'{func.__name__} cost time is {e - s:.8f}s')
        return result

    async def run_async(*args, **kwargs):
        s = time.perf_counter()
        result = func(*args, **kwargs)
        e = time.perf_counter()
        print(rf'{func.__name__} cost time is {e - s:.8f}s')
        return result

    if iscoroutinefunction(func):
        return run_async
    else:
        return run
