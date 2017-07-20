#include "calc_x2.h"
#include "cuda_utils.h"

static __global__ void kernelSumX2(float * x2, const float * x, int num_vec, int dim, int dim_aligned)
{
    for (int k = blockDim.y * blockIdx.x + threadIdx.y; k < num_vec; k += gridDim.x * blockDim.y)
    {
        float sum = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
        {
            float v = x[dim_aligned * k + d];
            sum += v * v;
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0)
            x2[k] = sum;
    }
}

static __global__ void kernelSumX2T(float * x2, const float * x, int num_vec, int num_vec_aligned, int dim)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            float v = x[num_vec_aligned * d + k];
            sum += v * v;
        }
        x2[k] = sum;
    }
}

static __global__ void kernelSumX2Sparse(float * x2, ohdSVM::csr_gpu x, int num_vec)
{
    for (int k = blockDim.y * blockIdx.x + threadIdx.y; k < num_vec; k += gridDim.x * blockDim.y)
    {
        float sum = 0;
        int beg = x.rowOffsets[k];
        int end = x.rowOffsets[k + 1];
        for (int d = beg + threadIdx.x; d < end; d += blockDim.x)
        {
            float v = x.values[d];
            sum += v * v;
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0)
            x2[k] = sum;
    }
}

static __global__ void kernelSumX2Sparse(float * x2, ohdSVM::jds_gpu x, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float sum = 0;
        int rowLen = x.rowLen[k];
        for (int d = 0; d < rowLen; d++)
        {
            float v = x.values[x.colStart[d] + k];
            sum += v * v;
        }
#ifdef JDS_PERMK
		x2[k] = sum;
#else
        x2[x.rowPerm[k]] = sum;
#endif
    }
}

static __global__ void kernelSumX2Sparse(float * x2, ohdSVM::ellrt_gpu x, int num_vec)
{
	extern __shared__ float shSum[];
	for (int k = blockDim.y * blockIdx.x + threadIdx.y; k < num_vec; k += gridDim.x * blockDim.y)
	{
		int sliceNum = k / x.sliceSize;
		int sliceRow = k % x.sliceSize;
		int threadStart = x.sliceStart[sliceNum] + blockDim.x * sliceRow;
		int rowLen = x.rowLen[k];
		float sum = 0;
		for (int b = 0; blockDim.x * b < rowLen; b++)
		{
			int d = blockDim.x * x.sliceSize * b + threadIdx.x;
			float v = x.values[threadStart + d];
			sum += v * v;
		}
		shSum[blockDim.x * threadIdx.y + threadIdx.x] = sum;
		__syncthreads();
		blockReduceSum(shSum + blockDim.x * threadIdx.y);
		if (threadIdx.x == 0)
			x2[k] = shSum[blockDim.x * threadIdx.y];
		__syncthreads();
	}
}

void ohdSVM::computeX2Dense(float * d_x2, const float * d_x, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    dim3 dimBlockSumX2(32, 8);
    kernelSumX2<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.y)), dimBlockSumX2>>>(d_x2, d_x, num_vec, dim, dim_aligned);
}

void ohdSVM::computeX2DenseT(float * d_x2, const float * d_xT, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    dim3 dimBlockSumX2(256);
    kernelSumX2T<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.x)), dimBlockSumX2>>>(d_x2, d_xT, num_vec, num_vec_aligned, dim);
}

void ohdSVM::computeX2Sparse(float * d_x2, const csr_gpu & x, int num_vec)
{
    dim3 dimBlockSumX2(32, 8);
    kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.y)), dimBlockSumX2>>>(d_x2, x, num_vec);
}

void ohdSVM::computeX2Sparse(float * d_x2, const jds_gpu & x, int num_vec)
{
    dim3 dimBlockSumX2(256);
    kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.x)), dimBlockSumX2>>>(d_x2, x, num_vec);
}

void ohdSVM::computeX2Sparse(float * d_x2, const ellrt_gpu & x, int num_vec)
{
	dim3 dimBlockSumX2(x.threadPerRow, 256 / x.threadPerRow);
	kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.y)), dimBlockSumX2, dimBlockSumX2.x * dimBlockSumX2.y * sizeof(float)>>>(d_x2, x, num_vec);
}
