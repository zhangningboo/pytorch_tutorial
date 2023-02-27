import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    __global__ void doublify(float *dst, const float *a)
    {
      int idx = threadIdx.x + threadIdx.y * 4;
      if (idx < 1000)
      {
        dst[idx] = a[idx] * 2;
      }
    }
    """)

if __name__ == '__main__':
    shape = (1000)
    dtype = np.int32
    my_input_array = np.random.randint(0, 10, shape).astype(dtype)

    mem_flags = cuda.mem_attach_flags.GLOBAL

    h_input = np.empty(shape, dtype)
    h_input[:] = my_input_array
    h_output = np.empty_like(h_input)

    u_input = cuda.managed_empty(shape, dtype, mem_flags=mem_flags)
    u_input[:] = my_input_array
    u_output = cuda.managed_empty(shape, dtype, mem_flags=mem_flags)

    my_gpu_function = mod.get_function('doublify')
    my_gpu_function(u_output, u_input, block=(4, 256, 1), grid=(1, 1, 1))

    # 没有人工拷贝，即cuda.memcpy_dtoh()时，需要显式同步
    pycuda.autoinit.context.synchronize()

    print(my_input_array)
    print(u_output)

    # autoinit的存在，显式free不需要
    # d_input.free()
    # d_output.free()
