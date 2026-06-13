__kernel void softmax_row(__global float *input, __global float *output, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;

    float max_val = -1e30f;
    for (int j = 0; j < N; j++) {
        max_val = fmax(max_val, input[row * N + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += exp(input[row * N + j] - max_val);
    }

    if (sum > 0.0f) {
        output[row * N + col] = exp(input[row * N + col] - max_val) / sum;
    } else {
        output[row * N + col] = 0.0f;
    }
}
