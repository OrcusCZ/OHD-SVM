#pragma once

__device__ size_t d_block_num_iter;
__device__ size_t d_total_num_iter;
__device__ float d_rho;
__device__ float d_diff;
__device__ int d_cache_rows_to_compute[MAX_WORKING_SET]; //must be >= WORKING_SET
__device__ int d_num_cache_rows_to_compute;
__device__ int d_updateGCnt[2];
__device__ int d_cacheUpdateCnt;

DEFINE_SYNC_BUFFERS(1); //for inter-blocks synchronization
