__kernel void matmul_naive(__global const float *A,
                           __global const float *B,
                           __global float *C,
                           int K, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
