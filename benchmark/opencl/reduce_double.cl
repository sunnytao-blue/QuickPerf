#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void reduce_sum_double(__global const double *input,
                                 __global double *output,
                                 __local double *local_sum,
                                 int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gsize = get_global_size(0);
    int group_size = get_local_size(0);

    double acc = 0.0;
    for (int i = gid; i < N; i += gsize) {
        acc += input[i];
    }
    local_sum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sum[lid] += local_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = local_sum[0];
    }
}
