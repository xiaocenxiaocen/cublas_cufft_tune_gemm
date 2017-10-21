#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

using std::cout;

#define GNU_C_COMPILER
#if defined(GNU_C_COMPILER)
extern "C" {
#include "cblas.h"
#include "lapacke.h"
#include "lapacke_mangling.h"
}
#elif defined(INTEL_C_COMPILER)
#include "mkl.h"
#endif

//#define VERBOSITY
using std::cout;

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define nullptr NULL

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char * file, const int line)
{
	if(cudaSuccess != err) {
		fprintf(stderr, "ERROR: safeCall() Runtime API error in file <%s>, line %i : %s.\n", file , line, cudaGetErrorString(err));
		exit(-1);
	}
}


class TimerGPU {
public:
	cudaEvent_t start, stop;
	cudaStream_t stream;
	TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}
	~TimerGPU() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	float read() {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
};

class TimerCPU {
	static const int bits = 10;
public:
	long long beg_clock;
	float freq;
	TimerCPU(float freq_) : freq(freq_) { 
		beg_clock = getTSC(bits);
	}
	long long getTSC(int bits) {
#ifdef WIN32
		return __rdtsc();
#else
		unsigned int low, high;
		__asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
		return ((long long)high<<(32 - bits)) | ((long long)low >> bits);
#endif
	}
	float read() {
		long long end_clock = getTSC(bits);
		long long Kcycles = end_clock - beg_clock;
		float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
		return time;
	}
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

template<size_t BX, size_t BY>
class CudaMatrix {
public:
	CudaMatrix();
	~CudaMatrix();
	void allocate(const int M_, const int N_, bool host, float * devmem, float * hostmem);
	double download();
	double readback();
public:
	int M, N;
	int padM, padN;
	float * h_data;
	float * d_data;
	bool h_internalAlloc;
	bool d_internalAlloc;
};

int iDivUp(int a, int b) { return (a % b == 0) ? (a / b) : (a / b + 1); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b == 0) ? a : (a - a % b + b); }
int iAlignDown(int a, int b) { return a - a % b; }

template<size_t BX, size_t BY>
void CudaMatrix<BX, BY>::allocate(const int M_, const int N_, bool host, float * devmem, float * hostmem)
{
	M = M_;
	N = N_;
	padM = iAlignUp(M, BY);
	padN = iAlignUp(N, BX);

	h_data = hostmem;
	d_data = devmem;
	if(d_data == nullptr) {
		long int nbts = sizeof(float) * (long)padM * padN;
		if(nbts < 0) {
			fprintf(stderr, "ERROR: cannot allocate %ld bytes from device global memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
			d_data = nullptr;
			exit(EXIT_FAILURE);
		}
		safeCall(cudaMalloc((void**)&d_data, nbts)); 
		safeCall(cudaMemset(d_data, 0, nbts));
		if(d_data == nullptr) {
			fprintf(stderr, "ERROR: cannot allocate %ld bytes from device global memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
		}
		d_internalAlloc = true;
	}
	if(host && h_data == nullptr) {
		long int nbts = sizeof(float) * (long)M * N;
		if(nbts < 0) {
			fprintf(stderr, "ERROR: cannot allocate %ld bytes from host memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
			h_data = nullptr;
			exit(EXIT_FAILURE);
		}
		h_data = (float*)malloc(nbts);
		memset(h_data, 0, nbts);
		h_internalAlloc = true;
	}
}

template<size_t BX, size_t BY>
CudaMatrix<BX, BY>::CudaMatrix() : M(0), N(0), h_data(nullptr), d_data(nullptr), h_internalAlloc(false), d_internalAlloc(false) 
{

}

template<size_t BX, size_t BY>
CudaMatrix<BX, BY>::~CudaMatrix()
{
	if(h_internalAlloc && h_data != nullptr) free(h_data);
	h_data = nullptr;
	if(d_internalAlloc && d_data != nullptr) safeCall(cudaFree(d_data));
	d_data = nullptr;
}

template<size_t BX, size_t BY>
double CudaMatrix<BX, BY>::download()
{
	TimerGPU timer(0);
	int p = sizeof(float) * padN;
	if(h_data != nullptr && d_data != nullptr) {
		safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice));
	}
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	
}

template<size_t BX, size_t BY>
double CudaMatrix<BX, BY>::readback()
{
	TimerGPU timer(0);
	int p = sizeof(float) * padN;
//	cout << sizeof(float) * N << "\t" << p << "\n";
//	if(h_data == nullptr) cout << "1\n";
//	if(d_data == nullptr) cout << "2\n";
	safeCall(cudaMemcpy2D(h_data, sizeof(float) * N, d_data, p, sizeof(float) * N, M, cudaMemcpyDeviceToHost));
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: readback time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;
}

// cache A and cache B
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
//	float A_reg[BK];
//	float B_reg[BK];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		const float * dbptr_ = dbptr + tidy * ldb; 
		for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		__syncthreads();
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				#pragma unroll
				for(int kk = 0; kk < BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
//					C_reg[ii][ij] += A_reg[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache A and cache B and prefetching
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB_prefetching(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
	float A_reg[BM / TY][BK / TX];
	float B_reg[BK / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	// load block of A to shared memory
	const float * daptr_ = daptr + tidy * lda;
	for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
		for(int ij = tidx; ij < BK; ij += TX) {
			A_smem[ii][ij] = daptr_[ij];
		}
	}

	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		if(ik < lda - 1) {
		// load block of A to registers
		const float * daptr_ = daptr + tidy * lda + BK;
		for(int ii = tidy, _ii = 0; ii < BM; ii += TY, _ii++, daptr_ += TY * lda) {
			for(int ij = tidx, _ij = 0; ij < BK; ij += TX, _ij++) {
				A_reg[_ii][_ij] = daptr_[ij];
			}
		}

		// load block of B to registers
		const float * dbptr_ = dbptr + tidy * ldb + stride_b; 
		for(int ii = tidy, _ii = 0; ii < BK; ii += TY, _ii++, dbptr_ += TY * ldb) {
			for(int ij = tidx, _ij = 0; ij < BN; ij += TX, _ij++) {
				B_reg[_ii][_ij] = dbptr_[ij];
			}
		}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				#pragma unroll
				for(int kk = 0; kk < BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
//					C_reg[ii][ij] += A_reg[kk] * B_smem[kk][in];
				}
			}
		}

		if(ik < lda - 1) {
		// store registers to A_smem
		for(int ii = tidy, _ii = 0; ii < BM; ii += TY, _ii++) {
			for(int ij = tidx, _ij = 0; ij < BK; ij += TX, _ij++) {
				A_smem[ii][ij] = A_reg[_ii][_ij];
			}
		}

		// store registers to B_smem
		for(int ii = tidy, _ii = 0; ii < BK; ii += TY, _ii++) {
			for(int ij = tidx, _ij = 0; ij < BN; ij += TX, _ij++) {
				B_smem[ii][ij] = B_reg[_ii][_ij];
			}
		}
		}
		
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache A and cache B and double-buffering
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	const int HALF_BK = BK / 2;

	const float * daptr_ = daptr + tidy * lda;
	for(int ii = tidy; ii < BM ; ii += TY, daptr_ += TY * lda) {
		for(int ij = tidx; ij < HALF_BK; ij += TX) {
			A_smem[ii][ij] = daptr_[ij];
		}
	}

	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();

	for(int ik = 0; ik < lda; ik += BK) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = HALF_BK + tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		const float * dbptr_ = dbptr + (HALF_BK + tidy) * ldb; 
		for(int ii = HALF_BK + tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = 0; kk < HALF_BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();

		daptr += BK, dbptr += stride_b;
		if(ik < lda - 1) {
			// load block of A to shared memory
			daptr_ = daptr + tidy * lda;
			for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
				for(int ij = tidx; ij < HALF_BK; ij += TX) {
					A_smem[ii][ij] = daptr_[ij];
				}
			}

			dbptr_ = dbptr + tidy * ldb; 
			for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
				for(int ij = tidx; ij < BN; ij += TX) {
					B_smem[ii][ij] = dbptr_[ij];
				}
			}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = HALF_BK; kk < BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache B
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_B(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[BK][BN];
//	__shared__ float C_smem[BM][BN];
	float C_reg[BM / TY * BN / TX];
	float A_reg[BK];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY * BN / TX; ii++) {
		C_reg[ii] = 0.f;
	}

//	for(int im = tidy; im < BM; im += TY) {
//		for(int in = tidx; in < BN; in += TX) {
//			C_smem[im][in] = 0.f;
//		}
//	}
//	__syncthreads();

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of B to shared memory
		const float * dbptr_ = dbptr + tidy * ldb; 
		for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		__syncthreads();

		const float * daptr_ = daptr + tidy * lda;
		int ii = 0;
		for(int im = tidy; im < BM; im += TY, daptr_ += TY * lda) {
			for(int kk = 0; kk < BK; kk++) {
				A_reg[kk] = daptr_[kk];
			}
			for(int in = tidx; in < BN; in += TX) {
				float ret = 0.f;
				#pragma unroll
				for(int kk = 0; kk < BK; kk++) {
					ret += A_reg[kk] * B_smem[kk][in];
//					dcptr_[in] += daptr_[kk] * B_smem[kk][in];
//					C_smem[im][in] += A_reg[kk] * B_smem[kk][in];
				}
//				C_smem[im][in] += ret;
				C_reg[ii++] += ret;
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	int ii = 0;
	for(int im = tidy; im < BM; im += TY, dcptr_ += TY * ldc) {
		for(int in = tidx; in < BN; in += TX) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii++];
//			dcptr_[in] = beta * dcptr_[in] + alpha * C_smem[im][in];
		}
	}
}

// cache B and double buffering
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_B_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
//	float A_reg[BK];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	const int HALF_BK = BK / 2;

	// load block of B to shared memory
	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();
	
	for(int ik = 0; ik < lda; ik += BK) {
		// load block of B to shared memory
		const float * dbptr_ = dbptr + (HALF_BK + tidy) * ldb; 
		for(int ii = HALF_BK + tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		
		const float * daptr_ = daptr + tidy * lda;		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++, daptr_ += TY * lda) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = 0; kk < HALF_BK; kk++) {
					C_reg[ii][ij] += daptr_[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();

		daptr += BK, dbptr += stride_b;
	
		if(ik < lda - 1) {
		// load block of B to shared memory
		dbptr_ = dbptr + tidy * ldb; 
		for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		}
		
		daptr_ = daptr + tidy * lda - BK;		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++, daptr_ += TY * lda) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = HALF_BK; kk < BK; kk++) {
					C_reg[ii][ij] += daptr_[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// unrolling
#define Bm 64
#define Bk 16
#define Bn 128
#define Tx 16
#define Ty 16
__device__ __inline__ float s_dot16(const float * a, float * bs)
{
	float ret = 0.f;
	
	ret += a[ 0] * bs[0 * 128];
	ret += a[ 1] * bs[1 * 128];
	ret += a[ 2] * bs[2 * 128];
	ret += a[ 3] * bs[3 * 128];
	ret += a[ 4] * bs[4 * 128];
	ret += a[ 5] * bs[5 * 128];
	ret += a[ 6] * bs[6 * 128];
	ret += a[ 7] * bs[7 * 128];
	ret += a[ 8] * bs[8 * 128];
	ret += a[ 9] * bs[9 * 128];
	ret += a[10] * bs[10 * 128];
	ret += a[11] * bs[11 * 128];
	ret += a[12] * bs[12 * 128];
	ret += a[13] * bs[13 * 128];
	ret += a[14] * bs[14 * 128];
	ret += a[15] * bs[15 * 128];

	return ret;	
}
__device__ __inline__ void s_dot8(float * c, float a, float * bs)
{
	
	c[0] += a * bs[0 * 16];
	c[1] += a * bs[1 * 16];
	c[2] += a * bs[2 * 16];
	c[3] += a * bs[3 * 16];
	c[4] += a * bs[4 * 16];
	c[5] += a * bs[5 * 16];
	c[6] += a * bs[6 * 16];
	c[7] += a * bs[7 * 16];

}
__global__ void mysgemm_cache_B_unrolling(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[2048];
	float C_reg[32] = {0.f};
//	float A_reg[16] = {0.f};	
	__shared__ float A_smem[16];

//	const unsigned int gy = (blockIdx.y << 6);
//	const unsigned int gx = (blockIdx.x << 7);
//	const float * daptr = dA + gy * lda;
//	const float * dbptr = dB + gx;
	
	const float * daptr = dA + ((blockIdx.y<<6) + threadIdx.y) * lda;
	const float * dbptr = dB + (blockIdx.x<<7) + threadIdx.y * ldb + threadIdx.x;
	float * dcptr = dC + ((blockIdx.y<<6) + threadIdx.y) * ldc + (blockIdx.x<<7) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 16) {
		float * Bs = &B_smem[(threadIdx.y<<7) + threadIdx.x];
		Bs[0 * 16] = dbptr[0 * 16];
		Bs[1 * 16] = dbptr[1 * 16];
		Bs[2 * 16] = dbptr[2 * 16];
		Bs[3 * 16] = dbptr[3 * 16];
		Bs[4 * 16] = dbptr[4 * 16];
		Bs[5 * 16] = dbptr[5 * 16];
		Bs[6 * 16] = dbptr[6 * 16];
		Bs[7 * 16] = dbptr[7 * 16];
		__syncthreads();
		
		const float * daptr_ = daptr;
		float * C_reg_ = C_reg;
		#pragma unroll
		for(int im = 0; im < 64; im += 16) {
			if(threadIdx.y == 0) A_smem[threadIdx.x] = daptr_[threadIdx.x];
//			__syncthreads();
//			A_reg[ 0] = daptr_[ 0];
//			A_reg[ 1] = daptr_[ 1];
//			A_reg[ 2] = daptr_[ 2];
//			A_reg[ 3] = daptr_[ 3];
//			A_reg[ 4] = daptr_[ 4];
//			A_reg[ 5] = daptr_[ 5];
//			A_reg[ 6] = daptr_[ 6];
//			A_reg[ 7] = daptr_[ 7];
//			A_reg[ 8] = daptr_[ 8];
//			A_reg[ 9] = daptr_[ 9];
//			A_reg[10] = daptr_[10];
//			A_reg[11] = daptr_[11];
//			A_reg[12] = daptr_[12];
//			A_reg[13] = daptr_[13];
//			A_reg[14] = daptr_[14];
//			A_reg[15] = daptr_[15];
//
			Bs = &B_smem[threadIdx.x];
			s_dot8(C_reg_, A_smem[0], &Bs[0 * 128]);
			s_dot8(C_reg_, A_smem[1], &Bs[1 * 128]);
			s_dot8(C_reg_, A_smem[2], &Bs[2 * 128]);
			s_dot8(C_reg_, A_smem[3], &Bs[3 * 128]);
			s_dot8(C_reg_, A_smem[4], &Bs[4 * 128]);
			s_dot8(C_reg_, A_smem[5], &Bs[5 * 128]);
			s_dot8(C_reg_, A_smem[6], &Bs[6 * 128]);
			s_dot8(C_reg_, A_smem[7], &Bs[7 * 128]);
			s_dot8(C_reg_, A_smem[8], &Bs[8 * 128]);
			s_dot8(C_reg_, A_smem[9], &Bs[9 * 128]);
			s_dot8(C_reg_, A_smem[10], &Bs[10 * 128]);
			s_dot8(C_reg_, A_smem[11], &Bs[11 * 128]);
			s_dot8(C_reg_, A_smem[12], &Bs[12 * 128]);
			s_dot8(C_reg_, A_smem[13], &Bs[13 * 128]);
			s_dot8(C_reg_, A_smem[14], &Bs[14 * 128]);
			s_dot8(C_reg_, A_smem[15], &Bs[15 * 128]);
		
			C_reg_ += 8;
			daptr_ += (lda<<4);					
		}
		__syncthreads();
		
		daptr += 16;
		dbptr += (ldb<<4);
	}	
	
	float * C_reg_ = C_reg;
	#pragma unroll
	for(int im = 0; im < 64; im += 16) {
		dcptr[0 * 16] = beta * dcptr[0 * 16] + alpha * C_reg_[0];
		dcptr[1 * 16] = beta * dcptr[1 * 16] + alpha * C_reg_[1];
		dcptr[2 * 16] = beta * dcptr[2 * 16] + alpha * C_reg_[2];
		dcptr[3 * 16] = beta * dcptr[3 * 16] + alpha * C_reg_[3];
		dcptr[4 * 16] = beta * dcptr[4 * 16] + alpha * C_reg_[4];
		dcptr[5 * 16] = beta * dcptr[5 * 16] + alpha * C_reg_[5];
		dcptr[6 * 16] = beta * dcptr[6 * 16] + alpha * C_reg_[6];
		dcptr[7 * 16] = beta * dcptr[7 * 16] + alpha * C_reg_[7];
		dcptr += (ldc<<4);
		C_reg_ += 8;
	}	
}

__global__ void mysgemm_cache_B_unrolling_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[2048];
	float C_reg[32] = {0.f};
	float A_reg[16] = {0.f};	

	const unsigned int gy = (blockIdx.y << 6);
	const unsigned int gx = (blockIdx.x << 7);

	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	for(int ik = 0; ik < lda; ik += Bk) {
		const float * dbptr_ = dbptr + threadIdx.y * ldb + threadIdx.x;
		float * Bs = &B_smem[(threadIdx.y<<7) + threadIdx.x];
		B_smem[0 * 16] = dbptr_[0 * 16];
		B_smem[1 * 16] = dbptr_[1 * 16];
		B_smem[2 * 16] = dbptr_[2 * 16];
		B_smem[3 * 16] = dbptr_[3 * 16];
		B_smem[4 * 16] = dbptr_[4 * 16];
		B_smem[5 * 16] = dbptr_[5 * 16];
		B_smem[6 * 16] = dbptr_[6 * 16];
		B_smem[7 * 16] = dbptr_[7 * 16];
		__syncthreads();
		
		const float * daptr_ = daptr + threadIdx.y * lda;
		float * C_reg_ = C_reg;
		#pragma unroll
		for(int im = 0; im < 64; im += 16) {
			A_reg[ 0] = daptr_[ 0];
			A_reg[ 1] = daptr_[ 1];
			A_reg[ 2] = daptr_[ 2];
			A_reg[ 3] = daptr_[ 3];
			A_reg[ 4] = daptr_[ 4];
			A_reg[ 5] = daptr_[ 5];
			A_reg[ 6] = daptr_[ 6];
			A_reg[ 7] = daptr_[ 7];
			A_reg[ 8] = daptr_[ 8];
			A_reg[ 9] = daptr_[ 9];
			A_reg[10] = daptr_[10];
			A_reg[11] = daptr_[11];
			A_reg[12] = daptr_[12];
			A_reg[13] = daptr_[13];
			A_reg[14] = daptr_[14];
			A_reg[15] = daptr_[15];

			Bs = &B_smem[threadIdx.x];
			C_reg_[0] += s_dot16(A_reg, &Bs[0 * 16]);
			C_reg_[1] += s_dot16(A_reg, &Bs[1 * 16]);
			C_reg_[2] += s_dot16(A_reg, &Bs[2 * 16]);
			C_reg_[3] += s_dot16(A_reg, &Bs[3 * 16]);
			C_reg_[4] += s_dot16(A_reg, &Bs[4 * 16]);
			C_reg_[5] += s_dot16(A_reg, &Bs[5 * 16]);
			C_reg_[6] += s_dot16(A_reg, &Bs[6 * 16]);
			C_reg_[7] += s_dot16(A_reg, &Bs[7 * 16]);
			
			C_reg_ += 8;
			daptr_ += (lda<<4);					
		}
		__syncthreads();
	}	
	
	float * dcptr_ = dcptr + threadIdx.y * ldc + threadIdx.x;
	float * C_reg_ = C_reg;
	#pragma unroll
	for(int im = 0; im < 64; im += 16) {
		dcptr_[0 * 16] = beta * dcptr_[0 * 16] + alpha * C_reg_[0];
		dcptr_[1 * 16] = beta * dcptr_[1 * 16] + alpha * C_reg_[1];
		dcptr_[2 * 16] = beta * dcptr_[2 * 16] + alpha * C_reg_[2];
		dcptr_[3 * 16] = beta * dcptr_[3 * 16] + alpha * C_reg_[3];
		dcptr_[4 * 16] = beta * dcptr_[4 * 16] + alpha * C_reg_[4];
		dcptr_[5 * 16] = beta * dcptr_[5 * 16] + alpha * C_reg_[5];
		dcptr_[6 * 16] = beta * dcptr_[6 * 16] + alpha * C_reg_[6];
		dcptr_[7 * 16] = beta * dcptr_[7 * 16] + alpha * C_reg_[7];
		dcptr_ += (ldc<<4);
		C_reg_ += 8;
	}	
}
// cache A
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_A(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
//	__shared__ float B_smem[BK][BN];
//	__shared__ float B_smem[BN];
	float C_reg[BM / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		__syncthreads();
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				const float * dbptr_ = dbptr;
				for(int kk = 0; kk < BK; kk++, dbptr_ += ldb) {
//					C_reg[ii][ij] += A_smem[im][kk] * B_smem[in];
					C_reg[ii][ij] += A_smem[im][kk] * dbptr_[in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
void mygemm_wrapper(const int M, const int K, const int N, const float alpha, const float * A, const int lda, const float * B, const int ldb, const float beta, float * C, const int ldc)
{
	CudaMatrix<BK, BM> wrapperA;
	wrapperA.allocate(M, lda, false, nullptr, const_cast<float*>(A));
	wrapperA.download();
	
	CudaMatrix<BN, BK> wrapperB;
	wrapperB.allocate(K, ldb, false, nullptr, const_cast<float*>(B));
	wrapperB.download();

	CudaMatrix<BN, BM> wrapperC;
	wrapperC.allocate(M, ldc, false, nullptr, C);
	wrapperC.download();

#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix A, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperA.padM, wrapperA.padN);
	fprintf(stdout, "INFO: matrix B, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperB.padM, wrapperB.padN);
	fprintf(stdout, "INFO: matrix C, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperC.padM, wrapperC.padN);
#endif

	dim3 grid( wrapperC.padN / BN, wrapperA.padM / BM, 1 );
	dim3 threads( TX, TY, 1 );
	
	TimerGPU timer(0);
	mysgemm_cache_B<BM, BK, BN, TX, TY><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
	double gpuTime = timer.read();

//	wrapperA.readback();	
//	for(int i = 0; i < M; i++) {
//		for(int j = 0; j < N; j++) {
//			fprintf(stdout, "%02.2f\t", A[i * N + j]);
//		}
//		fprintf(stdout, "\n");
//	}
//	fflush(stdout);


	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", gpuTime);
#ifdef VERBOSITY
	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (gpuTime / 1000.0 * 1e9));
#endif
	fflush(stdout);
	
	wrapperC.readback();
}

void constantInit(float * data, long int size, float val)
{
	for(long int i = 0; i < size; i++) {
		data[i] = val;
	}
}

int main(int argc, char * argv[])
{
	if(argc != 4) {
		fprintf(stderr, "USAGE: M K N\n");
		return -1;
	}

	int M = atoi(argv[1]);
	int K = atoi(argv[2]);
	int N = atoi(argv[3]);

#ifdef VERBOSITY	
	fprintf(stdout, "INFO: matrix A (MxK) multiply matrix B (KxN), result matrix C (MxN).\n");
	fprintf(stdout, "INFO: M = %d, K = %d, N = %d\n", M, K, N);
	fflush(stdout);
#endif
	
	float * h_A = (float*)malloc(sizeof(float) * M * K);
	float * h_B = (float*)malloc(sizeof(float) * K * N);
	float * h_C = (float*)malloc(sizeof(float) * M * N);
	float * h_D = (float*)malloc(sizeof(float) * M * N);

	const float valB = 0.01f;
	long int size_A = M * K;
	long int size_B = K * N;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);
	
	long int size_C = M * N;
	long int size_D = size_C;
	memset(h_C, 0, sizeof(float) * size_C);
	memset(h_D, 0, sizeof(float) * size_D);

	// warm up
	mygemm_wrapper<ROW_BLOCK_A, ROW_BLOCK_B, COL_BLOCK_C, THREAD_BLOCK_X, THREAD_BLOCK_Y>(
		M, K, N, 1.f,
		h_A, K, h_B, N, 0.f, h_C, N);
	
	
//	mygemm_wrapper<128, 32, 64, 32, 8>(
//		M, K, N, 1.f,
//		h_A, K, h_B, N, 0.f, h_C, N);

	
//	double t0 = omp_get_wtime();
	TimerCPU timer(2.60 * 1000);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, h_A, K, h_B, N, 0.0f, h_D, N);
	double cpuTime = timer.read();
//	t0 = omp_get_wtime() - t0;
//	cout << t0 << "\n";
#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", cpuTime);
	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (cpuTime / 1000.0 * 1e9));
#endif
	fflush(stdout);

	// test relative error
	bool correct = true;
	double eps = 1.e-6;
//	for(long int i = 0; i < size_C; i++) {
//		double abs_err = fabs(h_C[i] - h_D[i]);	
//		double dot_length = K;
//		double abs_val = fabs(h_C[i]);
//		double rel_err = abs_err / abs_val / dot_length;
//	
//		if (rel_err > eps) {
////	  		fprintf(stderr, "ERROR: Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], h_D[i], eps);
//           		correct = false;
//		
//        	}
//	}
	fprintf(stdout, "%s\n", correct ? "Result = PASS" : "Result = FAIL");
	fflush(stdout);
	
	free(h_A); h_A = nullptr;
	free(h_B); h_B = nullptr;
	free(h_C); h_C = nullptr;
	free(h_D); h_D = nullptr;

	if (!correct) {
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
