#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matmul_naive_double(__global const double *A,
                                   __global const double *B,
                                   __global double *C,
                                   int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
