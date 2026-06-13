#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void softmax_row_double(__global double *input, __global double *output, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;

    double max_val = -1e300;
    for (int j = 0; j < N; j++) {
        max_val = fmax(max_val, input[row * N + j]);
    }

    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        sum += exp(input[row * N + j] - max_val);
    }

    if (sum > 0.0) {
        output[row * N + col] = exp(input[row * N + col] - max_val) / sum;
    } else {
        output[row * N + col] = 0.0;
    }
}
