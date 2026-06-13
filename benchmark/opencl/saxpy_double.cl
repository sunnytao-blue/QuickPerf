#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void saxpy_double(__global const double *x, __global double *y, double alpha, int N) {
    int i = get_global_id(0);
    int stride = get_global_size(0);
    for (; i < N; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}
