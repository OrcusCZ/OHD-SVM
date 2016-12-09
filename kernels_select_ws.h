#pragma once

__global__ static void kernelPrepareSortI(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            v = y_ * g[k];
        else
            v = -FLT_MAX;
        valbuf[k] = v;
        idxbuf[k] = k;
    }
}

__global__ static void kernelPrepareSortJ(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            v = -y_ * g[k];
        else
            v = -FLT_MAX;
        valbuf[k] = v;
        idxbuf[k] = k;
    }
}

//blockSize >= NC
template<unsigned int blockSize, unsigned int NC, unsigned int numSortBlocks>
__global__ static void kernelFindNBest(const float * y, const float * g, const float * alpha, float C, int num_vec, int * ws_priority, float * aux_val, int * aux_idx, SYNC_BUFFER_DEF)
{
	const int sharedSize = 2 * blockSize;
	__shared__ float shValI[sharedSize];
	__shared__ float shValJ[sharedSize];
	__shared__ int shIdxI[sharedSize];
	__shared__ int shIdxJ[sharedSize];
	__shared__ int shTmpNum;

	int k = blockDim.x * blockIdx.x + threadIdx.x;
	float v, y_, a_, g_;
	if (k < num_vec)
	{
		y_ = y[k];
		a_ = alpha[k];
		g_ = g[k];

		if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
			shValI[threadIdx.x] = y_ * g_;
		else
			shValI[threadIdx.x] = -FLT_MAX;
		shIdxI[threadIdx.x] = k;

		if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
			shValJ[threadIdx.x] = -y_ * g_;
		else
			shValJ[threadIdx.x] = -FLT_MAX;
		shIdxJ[threadIdx.x] = k;

		if (threadIdx.x == 0) shTmpNum = 0;
	}
	else
	{
		shValI[threadIdx.x] = -FLT_MAX;
		shValJ[threadIdx.x] = -FLT_MAX;
	}
	__syncthreads();
	blockBitonicSort<true>(shIdxI, shValI);
	blockBitonicSort<true>(shIdxJ, shValJ);

	//main loop
	for (int koffset = gridDim.x * blockDim.x + blockDim.x * blockIdx.x; koffset < num_vec; koffset += gridDim.x * blockDim.x)
	{
		k = koffset + threadIdx.x;
		if (k < num_vec)
		{
			y_ = y[k];
			a_ = alpha[k];
			g_ = g[k];

			/////////////////////////
			//I part
#ifdef USE_DAIFLETCHER
			if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
				v = y_ - g_;
			else
				v = -FLT_MAX;
#else
			if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
				v = y_ * g_;
			else
				v = -FLT_MAX;
#endif

			//atomicInc and add new value
			if (v > shValI[NC - 1]) {
				int id = atomicAdd(&shTmpNum, 1);
				shValI[NC + id] = v;
				shIdxI[NC + id] = k;
			}
		}
		__syncthreads();


		//sort new-ones
		if (shTmpNum > 0) {
			int sortSize = NC * 2;
			while (shTmpNum + NC > sortSize)
				sortSize *= 2;
			if (threadIdx.x < sortSize - (shTmpNum + NC))
				shValI[shTmpNum + NC + threadIdx.x] = -FLT_MAX;
			__syncthreads();
			blockBitonicSortN<true>(shIdxI, shValI, sortSize);
			if (threadIdx.x == 0) shTmpNum = 0;
		}
		__syncthreads();

		if (k < num_vec)
		{
			/////////////////////////
			//J part
			if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
				v = -y_ * g_;
			else
				v = -FLT_MAX;

			//atomicInc and add new value
			if (v > shValJ[NC - 1]) {
				int id = atomicAdd(&shTmpNum, 1);
				shValJ[NC + id] = v;
				shIdxJ[NC + id] = k;
			}
		}
		__syncthreads();

		//sort new-ones
		if (shTmpNum > 0) {
			int sortSize = NC * 2;
			while (shTmpNum + NC > sortSize)
				sortSize *= 2;
			if (threadIdx.x < sortSize - (shTmpNum + NC))
				shValJ[shTmpNum + NC + threadIdx.x] = -FLT_MAX;
			__syncthreads();
			blockBitonicSortN<true>(shIdxJ, shValJ, sortSize);
			if (threadIdx.x == 0) shTmpNum = 0;
		}
		__syncthreads();
	}

	for (int i = threadIdx.x; i < NC; i += blockDim.x)
	{
		int gi = i + NC * blockIdx.x;
		int gj = gi + NC * gridDim.x;
		aux_idx[gi] = shIdxI[i];
		aux_val[gi] = shValI[i];
		aux_idx[gj] = shIdxJ[i];
		aux_val[gj] = shValJ[i];
	}

	WAIT_FOR_THE_FINAL_BLOCK;

	for (int i = threadIdx.x; i < NC; i += blockDim.x)
	{
		shIdxI[i] = aux_idx[i];
		shValI[i] = aux_val[i];
		shIdxJ[i] = aux_idx[i + NC * gridDim.x];
		shValJ[i] = aux_val[i + NC * gridDim.x];
	}
	for (int k = 1; k < gridDim.x; k++)
	{
		for (int i = threadIdx.x; i < NC; i += blockDim.x)
		{
			shIdxI[i + NC] = aux_idx[i + NC * k];
			shValI[i + NC] = aux_val[i + NC * k];
			shIdxJ[i + NC] = aux_idx[i + NC * (k + gridDim.x)];
			shValJ[i + NC] = aux_val[i + NC * (k + gridDim.x)];
		}
		__syncthreads();

		blockBitonicSortN<true>(shIdxI, shValI, 2 * NC);
		blockBitonicSortN<true>(shIdxJ, shValJ, 2 * NC);
	}

	if (shValI[NC - 1] <= -FLT_MAX || shValJ[NC - 1] <= -FLT_MAX)
		printf("[Error] Not enough elements found in FindNBest\n");

	for (int i = threadIdx.x; i < NC; i += blockDim.x)
	{
		int wsi = aux_idx[i] = shIdxI[i];
		int wsj = aux_idx[i + NC] = shIdxJ[i];
		ws_priority[wsi] = INT_MAX;
		ws_priority[wsj] = INT_MAX;
	}
}

template<unsigned int WS, unsigned int NC>
__global__ static void kernelFillWorkingSet(int * ws, const float * alpha, float C, int * ws_priority, int * new_ws)
{
    __shared__ int shWS[WS]; //old working set
    __shared__ int shNewWS[WS]; //new working set
    __shared__ int shWSPriority[WS];
    __shared__ int shNewWSPriority[WS];
    __shared__ float shAlpha[WS];

    for (int k = threadIdx.x; k < WS; k += blockDim.x)
    {
        int i = shWS[k] = ws[k];
        shWSPriority[k] = ws_priority[i];
        shAlpha[k] = alpha[i];
    }
    for (int k = threadIdx.x; k < NC * 2; k += blockDim.x)
    {
        shNewWS[k] = new_ws[k];
        shNewWSPriority[k] = 0;
    }
    __syncthreads();
    
    blockBitonicSort<false>(shWS, shWSPriority);

    int n = NC * 2;
    if (threadIdx.x == 0)
    {
        //free
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] > 0 && shAlpha[i] < C && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        //lower bound
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] <= 0 && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        //upper bound
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] >= C && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        if (n < WS)
            printf("[Error] Not enough elements to fill working set, this should never happen\n");
    }
    __syncthreads();

    for (int k = threadIdx.x; k < WS; k += blockDim.x)
    {
        int i = shNewWS[k];
        ws[k] = i;
        ws_priority[i] = shNewWSPriority[k];
    }
}