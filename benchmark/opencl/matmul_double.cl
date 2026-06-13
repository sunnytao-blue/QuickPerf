#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matmul_naive_double(__global const double *A,
                                   __global const double *B,
                                   __global double *C,
                                   int K, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
