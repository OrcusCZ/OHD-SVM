#pragma once

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWSKLocal(const int * KCacheRemapIdx, csr_gpu x, const int * ws, float * vec, int dim_aligned)
{
    int i = ws[blockIdx.y];
    if (KCacheRemapIdx[i] >= 0)
        return;

    int j = x.rowOffsets[i] + blockDim.x * blockIdx.x + threadIdx.x;
    while (j < x.rowOffsets[i + 1])
    {
        vec[dim_aligned * blockIdx.y + x.colInd[j]] = x.values[j];
        j += gridDim.x * blockDim.x;
    }
}

template<unsigned int WS, int BLOCK_SIZE>
__global__ static void kernelCalcKLocalDense_NN(float * KLocal, const float * K, const int * KCacheRemapIdx, const float * x, const float * x2, const float * y, const int * ws, float gamma, size_t num_vec_aligned, int dim, int dim_aligned)
{
    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ bool all_cached;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int colT = blockDim.x * blockIdx.x + threadIdx.y;
    int ws_idx = ws[row];
    int ws_idxJ = ws[colT];
    float sum = 0;
    if (threadIdx.x + threadIdx.y == 0)
        all_cached = true;
    __syncthreads();

    int cache_row = KCacheRemapIdx[ws_idx];
    if (cache_row < 0)
        all_cached = false;
    __syncthreads();

    if (!all_cached)
        for (int block = 0; block < dim; block += BLOCK_SIZE)
        {
            if (block + threadIdx.x < dim)
            {
                if (cache_row < 0)
                    shA[threadIdx.y][threadIdx.x] = x[dim_aligned * ws_idx + block + threadIdx.x];
                shB[threadIdx.x][threadIdx.y] = x[dim_aligned * ws_idxJ + block + threadIdx.x];
            }
            else
            {
                shA[threadIdx.y][threadIdx.x] = 0;
                shB[threadIdx.x][threadIdx.y] = 0;
            }
            __syncthreads();

            if (cache_row < 0)
                for (int d = 0; d < BLOCK_SIZE; d++)
                    sum += shA[threadIdx.y][d] * shB[d][threadIdx.x];
            __syncthreads();
        }

    if (row < WS && col < WS)
    {
        if (cache_row >= 0)
        {
            KLocal[WS * row + col] = K[(size_t)num_vec_aligned * cache_row + ws[col]];
        }
        else
        {
            sum = x2[ws_idx] + x2[ws[col]] - 2 * sum;
            KLocal[WS * row + col] = expf(-gamma * sum);
        }
    }
}

template<int WS, int NUM_WARPS>
__global__ static void kernelCalcKLocalSparse(float * KLocal, const float * K, const int * KCacheRemapIdx, csr_gpu x, const float * d_x2, const float * vec, const int * ws, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int ws_idx = ws[blockIdx.y];
    int cache_row = KCacheRemapIdx[ws_idx];

    int block = NUM_WARPS * blockIdx.x;
    int j = block + threadIdx.y;
    int ws_idxJ = ws[j];
    float sum = 0;
    if (cache_row < 0)
    {
        int end = x.rowOffsets[ws_idxJ] + x.rowLen[ws_idxJ];
        for (int d = x.rowOffsets[ws_idxJ] + threadIdx.x; d < end; d += warpSize)
        {
            sum += vec[dim_aligned * blockIdx.y + x.colInd[d]] * x.values[d];
        }
    }
    sum = warpReduceSum(sum);
    if (cache_row >= 0)
    {
        if (threadIdx.x < NUM_WARPS && threadIdx.y == 0)
            KLocal[WS * blockIdx.y + block + threadIdx.x] = K[(size_t)num_vec_aligned * cache_row + ws[block + threadIdx.x]];
    }
    else
    {
        if (threadIdx.x == 0)
        {
            sum = d_x2[ws_idx] + d_x2[ws_idxJ] - 2 * sum;
            KLocal[WS * blockIdx.y + j] = expf(-gamma * sum);
        }
    }
}

template<int WS, int NUM_WARPS>
__global__ static void kernelCalcKLocalSparsePerm(float * KLocal, const float * K, const int * KCacheRemapIdx, csr_gpu x, const unsigned int * rowPerm, const float * d_x2, const float * vec, const int * ws, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
	int ws_idx = ws[blockIdx.y];
	int cache_row = KCacheRemapIdx[ws_idx];

	int block = NUM_WARPS * blockIdx.x;
	int j = block + threadIdx.y;
	int ws_idxJ = ws[j];
	int permJ = rowPerm[ws_idxJ];
	float sum = 0;
	if (cache_row < 0)
	{
		int end = x.rowOffsets[permJ] + x.rowLen[permJ];
		for (int d = x.rowOffsets[permJ] + threadIdx.x; d < end; d += warpSize)
		{
			sum += vec[dim_aligned * blockIdx.y + x.colInd[d]] * x.values[d];
		}
	}
	sum = warpReduceSum(sum);
	if (cache_row >= 0)
	{
		if (threadIdx.x < NUM_WARPS && threadIdx.y == 0)
			KLocal[WS * blockIdx.y + block + threadIdx.x] = K[(size_t)num_vec_aligned * cache_row + ws[block + threadIdx.x]];
	}
	else
	{
		if (threadIdx.x == 0)
		{
			sum = d_x2[rowPerm[ws_idx]] + d_x2[permJ] - 2 * sum;
			KLocal[WS * blockIdx.y + j] = expf(-gamma * sum);
		}
	}
}

template<unsigned int WS>
__global__ static void kernelCopyKToLocal(const int * ws, const float * K, float * KLocal, int * KCacheRemapIdx, int num_vec_aligned)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    KLocal[WS * y + x] = K[(size_t)num_vec_aligned * KCacheRemapIdx[ws[y]] + ws[x]];
}

