#include "csr.h"
#include "cudaerror.h"
#include "cuda_utils.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

static __global__ void kernelCalcRowLen(unsigned int * rowLen, const unsigned int * rowOffsets, int numRows)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < numRows)
    {
        rowLen[k] = rowOffsets[k + 1] - rowOffsets[k];
    }
}

//make a GPU deep copy of a CPU csr matrix
void ohdSVM::makeCudaCsr(csr_gpu & x_gpu, const csr & x_cpu)
{
	x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;

	std::cout << "Allocating CSR structure of size "
		<< ((x_gpu.nnz * (sizeof(x_gpu.values) + sizeof(x_gpu.colInd))
			+ (x_gpu.numRows + 1) * sizeof(x_gpu.rowOffsets)
			+ x_gpu.numRows * sizeof(x_gpu.rowLen)) >> 20)
		<< " MB" << std::endl;

	assert_cuda(cudaMalloc((void **)&(x_gpu.values), x_gpu.nnz * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), x_gpu.nnz * sizeof(int)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.rowOffsets), (x_gpu.numRows+1) * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(int)));

	assert_cuda(cudaMemcpy(x_gpu.values, x_cpu.values, x_gpu.nnz * sizeof(float), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.colInd, x_cpu.colInd, x_gpu.nnz * sizeof(int), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.rowOffsets, x_cpu.rowOffsets, (x_gpu.numRows+1) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimBlock(256);
    dim3 dimGrid(getgriddim(x_gpu.numRows, dimBlock.x));
    kernelCalcRowLen<<<dimGrid, dimBlock>>>(x_gpu.rowLen, x_gpu.rowOffsets, x_gpu.numRows);
}

void ohdSVM::freeCudaCsr(csr_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
	cudaFree(x_gpu.rowOffsets);
    cudaFree(x_gpu.rowLen);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
	x_gpu.rowOffsets = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
}

template<typename T>
class IdxComparator
{
    T * v;
public:
    IdxComparator(T * v) : v(v) {}
    bool operator()(int i1, int i2)
    {
        return (*v)[i1] > (*v)[i2];
    }
};

void ohdSVM::makeCudaJds(jds_gpu & x_gpu, const csr & x_cpu)
{
    x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;

    std::vector<int> rowLen(x_cpu.numRows);
    std::adjacent_difference(x_cpu.rowOffsets + 1, x_cpu.rowOffsets + x_cpu.numRows + 1, rowLen.begin());
    x_gpu.maxRowLen = *std::max_element(rowLen.begin(), rowLen.end());

    std::vector<int> rowPerm(rowLen.size());
    std::iota(rowPerm.begin(), rowPerm.end(), 0);
    std::sort(rowPerm.begin(), rowPerm.end(), IdxComparator<std::vector<int>>(&rowLen));
    std::vector<int> rowLenSorted(rowLen.size());
    for (int i = 0; i < rowPerm.size(); i++)
        rowLenSorted[i] = rowLen[rowPerm[i]];

	std::cout << "Allocating JDS structure of size "
		<< ((x_gpu.nnz * (sizeof(x_gpu.values) + sizeof(x_gpu.colInd))
			+ x_gpu.numRows * (sizeof(x_gpu.rowLen) + sizeof(x_gpu.rowPerm))
			+ x_gpu.numCols * sizeof(x_gpu.colStart)) >> 20)
		<< " MB" << std::endl;

    assert_cuda(cudaMalloc((void **)&(x_gpu.values), x_gpu.nnz * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), x_gpu.nnz * sizeof(unsigned int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(unsigned int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowPerm), x_gpu.numRows * sizeof(unsigned int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.colStart), x_gpu.numCols * sizeof(unsigned int)));

    assert_cuda(cudaMemcpy(x_gpu.rowLen, &rowLenSorted[0], x_gpu.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.rowPerm, &rowPerm[0], x_gpu.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<float> values_jds(x_gpu.nnz);
    std::vector<unsigned int> colInd_jds(x_gpu.nnz);
    std::vector<unsigned int> colStart(x_gpu.maxRowLen);
    int out_idx = 0;
    for (int col = 0; col < x_gpu.maxRowLen; col++)
    {
        colStart[col] = out_idx;
        for (int row = 0; row < x_gpu.numRows; row++)
        {
            if (rowLenSorted[row] <= col)
                continue;
            int i = x_cpu.rowOffsets[rowPerm[row]] + col;
            values_jds[out_idx] = x_cpu.values[i];
            colInd_jds[out_idx] = x_cpu.colInd[i];
            out_idx++;
        }
    }

    assert_cuda(cudaMemcpy(x_gpu.colStart, &colStart[0], x_gpu.maxRowLen * sizeof(unsigned int), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.values, &values_jds[0], x_gpu.nnz * sizeof(float), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.colInd, &colInd_jds[0], x_gpu.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void ohdSVM::freeCudaJds(jds_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
    cudaFree(x_gpu.rowLen);
    cudaFree(x_gpu.rowPerm);
    cudaFree(x_gpu.colStart);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
    x_gpu.rowLen = NULL;
    x_gpu.rowPerm = NULL;
    x_gpu.colStart = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
    x_gpu.maxRowLen = 0;
}

void ohdSVM::makeCudaEllrt(ellrt_gpu & x_gpu, const csr & x_cpu, unsigned int sliceSize, unsigned int threadPerRow)
{
	x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;
	x_gpu.sliceSize = sliceSize;
	x_gpu.threadPerRow = threadPerRow;

	unsigned int numSlices = (x_gpu.numRows - 1) / sliceSize + 1;
	x_gpu.numSlices = numSlices;
	std::vector<int> rowLen(x_cpu.numRows);
	std::adjacent_difference(x_cpu.rowOffsets + 1, x_cpu.rowOffsets + x_cpu.numRows + 1, rowLen.begin());
	//x_gpu.maxRowLen = *std::max_element(rowLen.begin(), rowLen.end());

	size_t bufLen = 0;
	std::vector<int> sliceWidth(numSlices);
	for (int i = 0; i < numSlices; i++)
	{
		int maxLen = 0;
		for (int j = 0; j < sliceSize; j++)
		{
			int row = sliceSize * i + j;
			if (row >= x_gpu.numRows)
				break;
			if (maxLen < rowLen[row])
				maxLen = rowLen[row];
		}
		maxLen = ((maxLen - 1) / threadPerRow + 1) * threadPerRow;
		sliceWidth[i] = maxLen;
		bufLen += maxLen * sliceSize;
	}

	std::vector<float> values(bufLen, 0.f);
	std::vector<unsigned int> colInd(bufLen, 0);
	std::vector<size_t> sliceStart(numSlices);
	size_t curSliceStart = 0;
	for (int i = 0; i < numSlices; i++)
	{
		sliceStart[i] = curSliceStart;
		for (int b = 0; b < sliceWidth[i] / threadPerRow; b++)
		{
			int out_idx = curSliceStart + threadPerRow * sliceSize * b;
			for (int j = 0; j < sliceSize; j++)
			{
				int row = sliceSize * i + j;
				if (row >= x_gpu.numRows)
					break;
				for (int t = 0; t < threadPerRow; t++)
				{
					int col = threadPerRow * b + t;
					if (col < rowLen[row])
					{
						int idx = x_cpu.rowOffsets[row] + col;
						values[out_idx] = x_cpu.values[idx];
						colInd[out_idx] = x_cpu.colInd[idx];
					}
					out_idx++;
				}
			}
		}
		curSliceStart += sliceWidth[i] * (size_t)sliceSize;
	}
	std::cout << "Allocating EllR-T structure of size "
		<< ((bufLen * (sizeof(x_gpu.values) + sizeof(x_gpu.colInd))
			+ numSlices * sizeof(x_gpu.sliceStart)
			+ x_gpu.numRows * sizeof(x_gpu.rowLen)) >> 20)
		<< " MB" << std::endl;

	assert_cuda(cudaMalloc((void **)&(x_gpu.values), bufLen * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), bufLen * sizeof(unsigned int)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.sliceStart), numSlices * sizeof(size_t)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(unsigned int)));

	assert_cuda(cudaMemcpy(x_gpu.values, &values[0], bufLen * sizeof(float), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.colInd, &colInd[0], bufLen * sizeof(unsigned int), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.sliceStart, &sliceStart[0], numSlices * sizeof(size_t), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.rowLen, &rowLen[0], x_gpu.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void ohdSVM::freeCudaEllrt(ellrt_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
	cudaFree(x_gpu.sliceStart);
	cudaFree(x_gpu.rowLen);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
	x_gpu.rowLen = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
}