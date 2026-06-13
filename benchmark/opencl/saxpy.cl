__kernel void saxpy(__global const float *x, __global float *y, float alpha, int N) {
    int i = get_global_id(0);
    int stride = get_global_size(0);
    for (; i < N; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}
