# OHD-SVM
SVM training library using GPU optimized hierarchy decomposition algorithm.

## Compilation
CMake version 3.1 or higher is used to generate makefiles or Visual studio project files. The library is written in C++ and requires compiler with C++11 support. The only dependency is NVIDIA CUDA.

### Compilation steps
1. Download source codes (clone the repository).
2. Install NVIDIA CUDA Toolkit.
3. Use CMake to generate makefile / Visual Studio project files.
4. Compile using "make" or Visual Studio

## Usage
This project is not an executable application, it is a library exporting a function to train SVM model and must be linked with and called from a user application. Alternatively user can use [SVMbenchmark](https://github.com/OrcusCZ/SVMbenchmark) application to train SVM model using this library as is described at the end of this readme.

Application has to include header file `ohdSVM.h`.
All library functions and data types are in namespace `ohdSVM`.

Training data must be stored in union `ohdSVM::Data`.
```
union ohdSVM::Data
{
  const float * dense;
  const ohdSVM::csr * sparse;
}
```
Its member variable `dense` points to a linearized row major matrix of input data where each row is one training vector.
Rows offsets must be aligned to multiple of 32 floats.
Member variable `parse` points to an input data in CSR representation. Look at `ohdSVM::csr` structure definition for more details.

### Train function
The function to perform training is `ohdSVM::Train`.
```
bool ohdSVM::Train(float * alpha,
  float * rho,
  bool sparse,
  const Data & x,
  const float * y,
  size_t num_vec,
  size_t num_vec_aligned,
  size_t dim,
  size_t dim_aligned,
  float C,
  float gamma,
  float eps,
  int ws_size = 0)
```
This function returns true if training succeeds. Otherwise output variables are unchanged.

### Train function parameters
- alpha - output array for trained alphas, must be allocated by the caller and large enough to hold num_vec elements
- rho - output value
- sparse - true if training data is sparse
- x - training data. if dense then x.dense contains linearized matrix aligned to num_vec_aligned, if sparse  then x.sparse contains data in csr format
- y - training data labels, must contain values +1 or -1
- num_vec - number of training vectors
- num_vec_aligned - number of training vectors aligned up to the multiple of size of warp (32)
- dim training - data dimension
- dim_aligned - training data dimension aligned up to the multiple of size of warp (32)
- C - training parameter C
- gamma - RBF kernel parameter gamma
- eps - training threshold
- ws_size - working set size, 0 means to choose size automatically. it migt be necessary to lower the size for large datasets

## Integration with SVMbenchmark
SVMbenchmark is an application which can be used to traing SVM model using several SVM training implementations.
OHD-SVM is one of the supported implementations.
To use OHD-SVM with SVMbenchmark, do these steps:

1. Clone SVMbenchmark repository from [https://github.com/OrcusCZ/SVMbenchmark](https://github.com/OrcusCZ/SVMbenchmark).
2. Clone OHD-SVM repository into SVMbenchmark directory so you end up with folder structure `SVMbenchmark/OHD-SVM`.
3. Use CMake to create makefiles / project files. Cache variable COMPILE_WITH_OHDSVM needs to be set to ON (for ex. using cmake-gui and checking appropriate checkbox)
4. Other SVM implementations can be optionally enabled using other predefined cache variables.
5. Compile the SVMbenchmark project. OHD-SVM project is automatically included from SVMbenchmark CMake file and it is not necessary to run CMake or compile it separatedly.
6. Execute the benchmark using implementation number 16 to choose OHD-SVM (executable file argument `-i 16`). Please check SVMbenchmark readme for information how to execute the application.
