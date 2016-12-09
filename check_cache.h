#pragma once

//TODO: remove sparse arguments from cublas functions

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWS(const int * KCacheRowIdx, csr_gpu x, float * vec, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    int row = d_cache_rows_to_compute[blockIdx.y];
    int i = KCacheRowIdx[row];

    int j = x.rowOffsets[i] + blockDim.x * blockIdx.x + threadIdx.x;
    int end = x.rowOffsets[i + 1];
    while (j < end)
    {
        vec[dim_aligned * blockIdx.y + x.colInd[j]] = x.values[j];
        j += gridDim.x * blockDim.x;
    }
}

template<unsigned int WS, unsigned int TILE>
__global__ static void kernelCopyXTileT(float * xTile, const float * x, const int * KCacheRowIdx, size_t dim, size_t dim_aligned, size_t num_vec, size_t num_vec_aligned)
{
	__shared__ float tile[TILE][TILE + 1];
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x,
		yIndexO = blockDim.x * blockIdx.x + threadIdx.y,
        yIndex = blockDim.y * blockIdx.y + threadIdx.y,
        xIndexO = blockDim.y * blockIdx.y + threadIdx.x;
    int ws_size = d_num_cache_rows_to_compute;
    int row = d_cache_rows_to_compute[yIndex];
    if (xIndex < dim && yIndex < ws_size)
        tile[threadIdx.y][threadIdx.x] = x[dim_aligned * KCacheRowIdx[row] + xIndex];
    __syncthreads();
    if (xIndexO < ws_size && yIndexO < dim)
        xTile[WS * yIndexO + xIndexO] = tile[threadIdx.x][threadIdx.y];
    __syncthreads();
}

//block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS>
__global__ static void kernelFindCacheRow(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    if (threadIdx.x < WS)
    {
        if (KCacheRemapIdx[ws[threadIdx.x]] < 0)
            orderN = atomicAdd(&num, 1);
        else
            KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    }
    __syncthreads();

    for (int n = 0; n < num; n++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            shNIdx[n] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    if (orderN >= 0)
    {
        int cache_row = shNIdx[orderN];
        d_cache_rows_to_compute[orderN] = cache_row;
        int irow = KCacheRowIdx[cache_row];
        if (irow >= 0)
            KCacheRemapIdx[irow] = -1;
        KCacheRowIdx[cache_row] = ws[threadIdx.x];
        KCacheRemapIdx[ws[threadIdx.x]] = cache_row;
        KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//N * block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS, unsigned int N>
__global__ static void kernelFindCacheRowN(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN[N];
    for (int n = 0; n < N; n++)
        orderN[n] = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    for (int i = threadIdx.x, n = 0; i < WS; i += blockDim.x, n++)
    {
        if (KCacheRemapIdx[ws[i]] < 0)
            orderN[n] = atomicAdd(&num, 1);
        else
            KCacheRowPriority[KCacheRemapIdx[ws[i]]] = d_cacheUpdateCnt;
    }
    __syncthreads();

    for (int m = 0; m < num; m++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            shNIdx[m] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x, n = 0; n < N; i += blockDim.x, n++)
        if (orderN[n] >= 0)
        {
            int cache_row = shNIdx[orderN[n]];
            d_cache_rows_to_compute[orderN[n]] = cache_row;
            int irow = KCacheRowIdx[cache_row];
            if (irow >= 0)
                KCacheRemapIdx[irow] = -1;
            KCacheRowIdx[cache_row] = ws[i];
            KCacheRemapIdx[ws[i]] = cache_row;
            KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
        }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS>
__global__ static void kernelFindCacheRowKLocal(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows, const float * alphadiff)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    if (threadIdx.x < WS)
    {
        if (alphadiff[threadIdx.x] != 0)
        {
            if (KCacheRemapIdx[ws[threadIdx.x]] < 0)
                orderN = atomicAdd(&num, 1);
            else
                KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
        }
    }
    __syncthreads();

    for (int n = 0; n < num; n++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            shNIdx[n] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    if (orderN >= 0)
    {
        int cache_row = shNIdx[orderN];
        d_cache_rows_to_compute[orderN] = cache_row;
        int irow = KCacheRowIdx[cache_row];
        if (irow >= 0)
            KCacheRemapIdx[irow] = -1;
        KCacheRowIdx[cache_row] = ws[threadIdx.x];
        KCacheRemapIdx[ws[threadIdx.x]] = cache_row;
        KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//N * block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS, unsigned int N>
__global__ static void kernelFindCacheRowKLocalN(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows, const float * alphadiff)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN[N];
    for (int n = 0; n < N; n++)
        orderN[n] = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    for (int i = threadIdx.x, n = 0; i < WS; i += blockDim.x, n++)
    {
        if (alphadiff[i] != 0)
        {
            if (KCacheRemapIdx[ws[i]] < 0)
                orderN[n] = atomicAdd(&num, 1);
            else
                KCacheRowPriority[KCacheRemapIdx[ws[i]]] = d_cacheUpdateCnt;
        }
    }
    __syncthreads();

    for (int m = 0; m < num; m++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            shNIdx[m] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x, n = 0; n < N; i += blockDim.x, n++)
        if (orderN[n] >= 0)
        {
            int cache_row = shNIdx[orderN[n]];
            d_cache_rows_to_compute[orderN[n]] = cache_row;
            int irow = KCacheRowIdx[cache_row];
            if (irow >= 0)
                KCacheRemapIdx[irow] = -1;
            KCacheRowIdx[cache_row] = ws[i];
            KCacheRemapIdx[ws[i]] = cache_row;
            KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
        }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

__global__ static void kernelCublasFinalize(float * K, const float * KTile, const float * x2, const int * KCacheRowIdx, size_t num_vec, size_t num_vec_aligned, float gamma)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_vec && k < d_num_cache_rows_to_compute)
    {
        int row = d_cache_rows_to_compute[k];
        int j = KCacheRowIdx[row];
        size_t idx = num_vec_aligned * k + i;
        float s = KTile[idx];
        s = x2[j] + x2[i] - 2 * s;
        K[num_vec_aligned * row + i] = expf(-gamma * s);
    }
}

//<4,4>
//block dim: 32 x number of warps
template<int TILE_X, int TILE_Y, int NUM_WARPS>
__global__ static void kernelCalcCacheDense(float * d_K, int * d_KCacheRowIdx, const float * d_x, const float * d_x2, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y * TILE_Y)
        return;
    int num_y = TILE_Y;
    if (d_num_cache_rows_to_compute < (blockIdx.y + 1) * TILE_Y)
        num_y = d_num_cache_rows_to_compute % TILE_Y;
    __shared__ float shOut[TILE_X * TILE_Y * NUM_WARPS];
    int row[TILE_Y];
    int i[TILE_Y];
    for (int y = 0; y < TILE_Y; y++)
    {
        row[y] = d_cache_rows_to_compute[blockIdx.y * TILE_Y + y];
        i[y] = d_KCacheRowIdx[row[y]];
    }
    int block = NUM_WARPS * TILE_X * blockIdx.x;

    //calculate cache matrix row [row], original index is [i]
    while (block < num_vec)
    {
        int j = block + threadIdx.y * TILE_X;
        float sum[TILE_Y][TILE_X] = {0};
        if (j + TILE_X - 1 < num_vec)
        {
            for (int d = threadIdx.x; d < dim; d += warpSize)
            {
#pragma unroll
                for (int y = 0; y < TILE_Y; y++)
#pragma unroll
                    for (int x = 0; x < TILE_X; x++)
                        sum[y][x] += d_x[dim_aligned * i[y] + d] * d_x[dim_aligned * (j + x) + d];
            }
        }
        else
        {
            for (int d = threadIdx.x; d < dim; d += warpSize)
            {
#pragma unroll
                for (int x = 0; x < TILE_X; x++)
                    if (j + x < num_vec)
#pragma unroll
                        for (int y = 0; y < TILE_Y; y++)
                            sum[y][x] += d_x[dim_aligned * i[y] + d] * d_x[dim_aligned * (j + x) + d];
            }
        }
#pragma unroll
        for (int y = 0; y < TILE_Y; y++)
#pragma unroll
            for (int x = 0; x < TILE_X; x++)
            {
                float s = warpReduceSum(sum[y][x]);
                if (threadIdx.x == 0 && j + x < num_vec && y < num_y)
                {
                    s = d_x2[i[y]] + d_x2[j + x] - 2 * s;
                    shOut[NUM_WARPS * TILE_X * y + TILE_X * threadIdx.y + x] = expf(-gamma * s);
                }
            }
        __syncthreads();
        for (int x = threadIdx.x; x < NUM_WARPS * TILE_X && block + threadIdx.x < num_vec; x += blockDim.x)
        {
            for (int y = threadIdx.y; y < num_y; y += blockDim.y)
            {
                d_K[(size_t)num_vec_aligned * row[y] + block + x] = shOut[NUM_WARPS * TILE_X * y + x];
            }
        }
        __syncthreads();

        block += gridDim.x * blockDim.y * TILE_X;
    }
}

template<int TILE_X, int TILE_Y, int NUM_WARPS>
__global__ static void kernelCalcCacheSparse(float * K, int * d_KCacheRowIdx, csr_gpu x, const float * d_x2, const float * vec, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y * TILE_Y)
        return;
    int num_y = TILE_Y;
    if (d_num_cache_rows_to_compute < (blockIdx.y + 1) * TILE_Y)
        num_y = d_num_cache_rows_to_compute % TILE_Y;
    __shared__ float shOut[TILE_X * TILE_Y * NUM_WARPS];
    int row[TILE_Y];
    int i[TILE_Y];
    for (int y = 0; y < TILE_Y; y++)
    {
        row[y] = d_cache_rows_to_compute[blockIdx.y * TILE_Y + y];
        i[y] = d_KCacheRowIdx[row[y]];
    }

    int block = NUM_WARPS * TILE_X * blockIdx.x;
    while (block < num_vec)
    {
        int j = block + threadIdx.y * TILE_X;
        float sum = 0;
        if (j < num_vec)
        {
            int end = x.rowOffsets[j] + x.rowLen[j];
            for (int d = x.rowOffsets[j] + threadIdx.x; d < end; d += warpSize)
            {
                sum += vec[dim_aligned * blockIdx.y + x.colInd[d]] * x.values[d];
            }
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0 && j < num_vec)
        {
            sum = d_x2[i[0]] + d_x2[j] - 2 * sum;
            K[(size_t)num_vec_aligned * row[0] + j] = expf(-gamma * sum);
        }
        __syncthreads();
        block += gridDim.x * blockDim.y * TILE_X;
    }
}

__global__ static void kernelCalcCacheSparse(float * K, int * d_KCacheRowIdx, jds_gpu x, const float * d_x2, const float * vec, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    int row = d_cache_rows_to_compute[blockIdx.y];
    int i = d_KCacheRowIdx[row];
    float x2i = d_x2[i];

    int block = blockDim.x * blockIdx.x;
    while (block < num_vec)
    {
        int r = block + threadIdx.x;
        if (r < num_vec)
        {
            float sum = 0;
            int rowLen = x.rowLen[r];
            for (int d = 0; d < rowLen; d++)
            {
                int i = x.colStart[d] + r;
                sum += vec[dim_aligned * blockIdx.y + x.colInd[i]] * x.values[i];
            }
            sum = x2i + d_x2[x.rowPerm[r]] - 2 * sum;
            K[(size_t)num_vec_aligned * row + x.rowPerm[r]] = expf(-gamma * sum);
        }
        block += gridDim.x * blockDim.x;
    }
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCache(bool sparse, const int * d_workingset, float * d_x, const float * d_x2, const csr_gpu & csr_data_gpu, const jds_gpu & jds_data_gpu, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    else
        kernelFindCacheRow<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    if (sparse)
    {
        assert_cuda(cudaMemset(d_denseVec, 0, dim_aligned * WS * sizeof(float)));
        dim3 dimBlock(256);
        dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)), WS);
        kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, csr_data_gpu, d_denseVec, dim_aligned);
        dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
		kernelCalcCacheSparse << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, jds_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    else
    {
        const int TILE_X = 4;
        const int TILE_Y = 8;
        const int NUM_WARPS = 4;
        dim3 dimBlock(32, NUM_WARPS);
        dim3 dimGrid(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
		kernelCalcCacheDense<TILE_X, TILE_Y, NUM_WARPS><<<dimGrid, dimBlock>>>(d_K, d_KCacheRowIdx, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheKLocal(bool sparse, const int * d_workingset, float * d_x, const float * d_x2, const csr_gpu & csr_data_gpu, const jds_gpu & jds_data_gpu, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_alphadiff, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowKLocalN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    else
        kernelFindCacheRowKLocal<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    if (sparse)
    {
        assert_cuda(cudaMemset(d_denseVec, 0, dim_aligned * WS * sizeof(float)));
        dim3 dimBlock(256);
        dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)), WS);
        kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, csr_data_gpu, d_denseVec, dim_aligned);
        dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
		kernelCalcCacheSparse << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, jds_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    else
    {
        const int TILE_X = 4;
        const int TILE_Y = 8;
        const int NUM_WARPS = 4;
        dim3 dimBlock(32, NUM_WARPS);
        dim3 dimGrid(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
		kernelCalcCacheDense<TILE_X, TILE_Y, NUM_WARPS><<<dimGrid, dimBlock>>>(d_K, d_KCacheRowIdx, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheCublas(const int * d_workingset, float * d_x, float * d_xT, float * d_xTile, const float * d_x2, float * d_K, float * d_KTile, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma, cublasHandle_t cublas)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    else
        kernelFindCacheRow<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);

    const int TILE = 16;
    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid(getgriddim<size_t>(dim, dimBlock.x), getgriddim<size_t>(WS, dimBlock.y));
    kernelCopyXTileT<WS, TILE><<<dimGrid, dimBlock>>>(d_xTile, d_x, d_KCacheRowIdx, dim, dim_aligned, num_vec, num_vec_aligned);

    float alpha = 1,
          beta = 0;
    assert_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, num_vec, WS, dim, &alpha, d_xT, num_vec_aligned, d_xTile, WS, &beta, d_KTile, num_vec_aligned));

    dimGrid.x = getgriddim<size_t>(num_vec, dimBlock.x);
    kernelCublasFinalize<<<dimGrid, dimBlock>>>(d_K, d_KTile, d_x2, d_KCacheRowIdx, num_vec, num_vec_aligned, gamma);
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheCublasKLocal(const int * d_workingset, float * d_x, float * d_xT, float * d_xTile, const float * d_x2, float * d_K, float * d_KTile, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, const float * d_alphadiff, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma, cublasHandle_t cublas)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowKLocalN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    else
        kernelFindCacheRowKLocal<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    int num_cache_rows_to_compute;
    assert_cuda(cudaMemcpyFromSymbol(&num_cache_rows_to_compute, d_num_cache_rows_to_compute, sizeof(int)));
    if (num_cache_rows_to_compute <= 0)
        return;

    const int TILE = 16;
    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid(getgriddim<size_t>(dim, dimBlock.x), getgriddim<size_t>(num_cache_rows_to_compute, dimBlock.y));
    kernelCopyXTileT<WS, TILE><<<dimGrid, dimBlock>>>(d_xTile, d_x, d_KCacheRowIdx, dim, dim_aligned, num_vec, num_vec_aligned);

    float alpha = 1,
          beta = 0;
    assert_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, num_vec, num_cache_rows_to_compute, dim, &alpha, d_xT, num_vec_aligned, d_xTile, WS, &beta, d_KTile, num_vec_aligned));

    dimGrid.x = getgriddim<size_t>(num_vec, dimBlock.x);
    kernelCublasFinalize<<<dimGrid, dimBlock>>>(d_K, d_KTile, d_x2, d_KCacheRowIdx, num_vec, num_vec_aligned, gamma);
}
