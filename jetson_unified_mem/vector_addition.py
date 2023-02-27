import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from count_time import timeit

shape = (1 << 24)
dtype = np.float32

my_input_array_a = np.random.randint(0, 10, shape).astype(dtype)
my_input_array_b = np.random.randint(0, 10, shape).astype(dtype)

kernel = """
extern "C"
__global__ void mykern(float *dst, const float *a, const float *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        dst[i] = a[i] + b[i];
    }
}
"""

mod = SourceModule(kernel)
grid = (512, 1)
block = (512, 1, 1)


@timeit
def original_dGPU():
    h_a = np.empty(shape, dtype)
    h_b = np.empty(shape, dtype)
    h_c = np.empty(shape, dtype)

    h_a[:] = my_input_array_a
    h_b[:] = my_input_array_b

    d_a = cuda.mem_alloc(h_a.nbytes)
    d_b = cuda.mem_alloc(h_a.nbytes)
    d_c = cuda.mem_alloc(h_a.nbytes)

    cuda.memcpy_htod(dest=d_a, src=h_a)
    cuda.memcpy_htod(dest=d_b, src=h_b)

    func = mod.get_function('mykern')
    func(d_c, d_a, d_b, np.array([shape]), grid=grid, block=block)

    cuda.memcpy_dtoh(dest=h_c, src=d_c)
    print(h_a[:10])
    print(h_b[:10])
    print(h_c[:10])


@timeit
def um_dGPU():
    mem_flags = cuda.mem_attach_flags.GLOBAL

    u_a = cuda.managed_empty(shape, dtype, mem_flags=mem_flags)
    u_b = cuda.managed_empty_like(u_a, mem_flags=mem_flags)
    u_c = cuda.managed_empty_like(u_a, mem_flags=mem_flags)

    u_a[:] = my_input_array_a
    u_b[:] = my_input_array_b

    func = mod.get_function('mykern')
    func(u_c, u_a, u_b, np.array([shape]), grid=grid, block=block)
    pycuda.autoinit.context.synchronize()
    print(u_a[:10])
    print(u_b[:10])
    print(u_c[:10])


@timeit
def jetson_GPU():
    # all same as um_dGPU, but run on Jetson
    return um_dGPU()


@timeit
def pinned_mem():
    a = cuda.pagelocked_empty(shape, dtype)
    b = cuda.pagelocked_empty_like(a)
    c = cuda.pagelocked_empty_like(a)

    a[:] = my_input_array_a
    b[:] = my_input_array_b

    d_a = np.intp(a.base.get_device_pointer())
    d_b = np.intp(b.base.get_device_pointer())
    d_c = np.intp(c.base.get_device_pointer())

    func = mod.get_function('mykern')
    func(d_c, d_a, d_b, np.array([shape]), grid=grid, block=block)
    pycuda.autoinit.context.synchronize()
    print(a[:10])
    print(b[:10])
    print(c[:10])


if __name__ == '__main__':
    "https://zhuanlan.zhihu.com/p/486130961"
    original_dGPU()
    um_dGPU()
    jetson_GPU()
    pinned_mem()
