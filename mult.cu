#include <stdio.h>
#include <math.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 8

// Forward declaration of the matrix multiplication kernel
__global__
void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// main function allocating the matrices and calling the multiplication function
int main(int argc, char** argv)
{
    int m=1000, n=1000;
    int row, col;
    float sum = 0;
    float tmp;
    Matrix A, B, C;
    A.width = m; A.height = n;
    B.width = m; B.height = n;
    C.width = m; C.height = n;
    A.elements = (float*) calloc ((A.width)*(A.height), sizeof(float));
    B.elements = (float*) calloc ((B.width)*(B.height), sizeof(float));
    C.elements = (float*) calloc ((C.width)*(C.height), sizeof(float));

    col = 0;
    for (row=0; row<A.width; row++) {
        //for (col=0; col<A.height; col++) {
            A.elements[row*A.width+col] = 1;
            B.elements[row*A.width+col] = 2;
            col++;
        //}
    }


    MatMul(A, B, C);

    for (row=0; row<C.width; row++) {
        tmp = 0;
        for (col=0; col<C.height; col++) {
            tmp += C.elements[row*C.width+col];
        }
        sum += tmp;
    }
    printf("Sum of all elements of C is: %f\n", sum);
    
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
