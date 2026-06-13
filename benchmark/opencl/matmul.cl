__kernel void matmul_naive(__global const float *A,
                           __global const float *B,
                           __global float *C,
                           int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
