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

    h_input = np.empty(shape, dtype)
    h_input[:] = my_input_array
    h_output = np.empty_like(h_input)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    cuda.memcpy_htod(dest=d_input, src=h_input)

    my_gpu_function = mod.get_function('doublify')
    my_gpu_function(d_output, d_input, block=(4, 256, 1), grid=(1, 1, 1))

    cuda.memcpy_dtoh(dest=h_output, src=d_output)

    print(my_input_array)
    print(h_output)

    # autoinit的存在，显式free不需要
    # d_input.free()
    # d_output.free()
