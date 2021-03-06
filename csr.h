#pragma once

namespace ohdSVM
{
	struct csr {
		unsigned int nnz;
		unsigned int numRows;
		unsigned int numCols;
		float *values;
		unsigned int *colInd;
		unsigned int *rowOffsets;
	};

	struct csr_gpu {
		unsigned int nnz;
		unsigned int numRows;
		unsigned int numCols;
		float *values;
		unsigned int *colInd;
		unsigned int *rowOffsets;
		unsigned int *rowLen;
	};

	struct jds_gpu {
		unsigned int nnz;
		unsigned int numRows;
		unsigned int numCols;
		unsigned int maxRowLen;
		float *values;
		unsigned int *colInd;
		unsigned int *rowLen;
		unsigned int *rowPerm;
		unsigned int *colStart;
	};

	struct ellrt_gpu {
		unsigned int nnz;
		unsigned int numRows;
		unsigned int numCols;
		float *values;
		unsigned int *colInd;
		size_t *sliceStart;
		unsigned int *rowLen;
		unsigned int numSlices;
		unsigned int sliceSize;
		unsigned int threadPerRow;
	};

    void makeCudaCsr(csr_gpu & x_gpu, const csr & x_cpu);
    void freeCudaCsr(csr_gpu & x_gpu);
    void makeCudaJds(jds_gpu & x_gpu, const csr & x_cpu);
    void freeCudaJds(jds_gpu & x_gpu);
	void makeCudaEllrt(ellrt_gpu & x_gpu, const csr & x_cpu, unsigned int sliceSize, unsigned int threadPerRow);
	void freeCudaEllrt(ellrt_gpu & x_gpu);
}
