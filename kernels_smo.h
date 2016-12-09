#pragma once

//storing only Ith row of K matrix in shared memory
//num threads = WS
template<unsigned int WS>
__global__ static void kernelSMO1Block(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
    __shared__ int shIdx[WS];
    __shared__ float shI[WS];
    __shared__ float shJ[WS];
    __shared__ float shLambda1, shLambda2;

    int wsi = d_ws[threadIdx.x];
    float y = d_y[wsi];
    float g = d_g[wsi];
    float a = d_alpha[wsi];
    float aold = a;
    __syncthreads();
	
	float eps2;

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            shI[threadIdx.x] = y * g;
        else
            shI[threadIdx.x] = -FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            shJ[threadIdx.x] = -y * g;
        else
            shJ[threadIdx.x] = -FLT_MAX;
        int wsI = blockMaxReduce(shI, shIdx);
        __syncthreads();
        float KI = d_K[WS * wsI + threadIdx.x];
        int wsJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[wsI];

        float diff = vI + shJ[wsJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
            d_alpha[wsi] = a;
            d_alphadiff[threadIdx.x] = -(a - aold) * y;
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[wsJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        if (((y > 0 && a > 0) || (y < 0 && a < C)) && vI > y * g)
        {
            float den = 1 + 1 - 2 * KI;
            float val = vI - y * g;
            shI[threadIdx.x] = val * val / den;
        }
        else
            shI[threadIdx.x] = -FLT_MAX;
        wsJ = blockMaxReduce(shI, shIdx);
        float KJ = d_K[WS * wsJ + threadIdx.x];

        //update alpha
        if (threadIdx.x == wsI)
            shLambda1 = y > 0 ? C - a : a;
        if (threadIdx.x == wsJ)
            shLambda2 = min(y > 0 ? a : C - a, (vI + shJ[wsJ]) / (1 + 1 - 2 * KI));
        __syncthreads();
        float l = min(shLambda1, shLambda2);
        
        if (threadIdx.x == wsI)
            a += l * y;
        if (threadIdx.x == wsJ)
            a -= l * y;

        //update g
        g += l * y * (KJ - KI);
	}
}

//storing only Ith row of K matrix in shared memory
//each thread processes several elements
//num threads = WS / N
template<unsigned int WS, unsigned int N>
__global__ static void kernelSMO1BlockN(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
	__shared__ float shKI[WS];
    __shared__ int shIdx[WS/N];
    __shared__ float shI[WS/N];
    __shared__ float shJ[WS/N];
    __shared__ int shIdxThread[WS/N];
    __shared__ float shLambda;

    int wsi[N];
    float y[N],
          g[N],
          a[N],
          aold[N];
#pragma unroll
    for (int n = 0; n < N; n++)
    {
        wsi[n] = d_ws[blockDim.x * n + threadIdx.x];
        y[n] = d_y[wsi[n]];
        g[n] = d_g[wsi[n]];
        aold[n] = a[n] = d_alpha[wsi[n]];
    }
    __syncthreads();
	
	float eps2;

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        float maxI = -FLT_MAX;
        float maxJ = -FLT_MAX;
        int maxIidx;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            if ((y[n] > 0 && a[n] < C) || (y[n] < 0 && a[n] > 0))
            {
                float v = y[n] * g[n];
                if (v > maxI)
                {
                    maxI = v;
                    maxIidx = n;
                }
            }
            if ((y[n] > 0 && a[n] > 0) || (y[n] < 0 && a[n] < C))
            {
                float v = -y[n] * g[n];
                if (v > maxJ)
                    maxJ = v;
            }
        }
        shI[threadIdx.x] = maxI;
        shJ[threadIdx.x] = maxJ;
        shIdxThread[threadIdx.x] = maxIidx;
        int thI = blockMaxReduce(shI, shIdx);
        int wsI = blockDim.x * shIdxThread[thI] + thI;
#pragma unroll
        for (int n = 0; n < N; n++)
            shKI[blockDim.x * n + threadIdx.x] = d_K[WS * wsI + blockDim.x * n + threadIdx.x];
        int thJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[thI];

        float diff = vI + shJ[thJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
#pragma unroll
            for (int n = 0; n < N; n++)
            {
                d_alpha[wsi[n]] = a[n];
                d_alphadiff[blockDim.x * n + threadIdx.x] = -(a[n] - aold[n]) * y[n];
            }
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[thJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        maxJ = -FLT_MAX;
        int maxJidx;
        for (int n = 0; n < N; n++)
        {
            if (((y[n] > 0 && a[n] > 0) || (y[n] < 0 && a[n] < C)) && vI > y[n] * g[n])
            {
                float den = 1 + 1 - 2 * shKI[blockDim.x * n + threadIdx.x];
                float val = vI - y[n] * g[n];
                val = val * val / den;
                if (val > maxJ)
                {
                    maxJ = val;
                    maxJidx = n;
                }
            }
        }
        shI[threadIdx.x] = maxJ;
        shIdxThread[threadIdx.x] = maxJidx;
        thJ = blockMaxReduce(shI, shIdx);
        int wsJ = blockDim.x * shIdxThread[thJ] + thJ;
        float KJ[N];
#pragma unroll
        for (int n = 0; n < N; n++)
            KJ[n] = d_K[WS * wsJ + blockDim.x * n + threadIdx.x];

        //update alpha
        if (threadIdx.x == thI)
            shLambda = y[maxIidx] > 0 ? C - a[maxIidx] : a[maxIidx];
        __syncthreads();
        if (threadIdx.x == thJ)
        {
            float l = min(shLambda, y[maxJidx] > 0 ? a[maxJidx] : C - a[maxJidx]);
            shLambda = min(l, (vI - y[maxJidx] * g[maxJidx]) / (1 + 1 - 2 * shKI[wsJ]));
        }
        
        __syncthreads();
        if (threadIdx.x == thI)
            a[maxIidx] += shLambda * y[maxIidx];
        if (threadIdx.x == thJ)
            a[maxJidx] -= shLambda * y[maxJidx];

        //update g
#pragma unroll
        for (int n = 0; n < N; n++)
            g[n] += shLambda * y[n] * (KJ[n] - shKI[blockDim.x * n + threadIdx.x]);
	}
}
