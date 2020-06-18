#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

void MatMulSerial(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0;
    int row,col,ks;
    for (row=0;row<A.width;row++) {
        for (col=0;col<A.height;col++) {
            for (ks=0; ks<A.width; ks++) {
                Cvalue += A.elements[row * A.width + ks] * B.elements[ks * B.width + col];
            }
            C.elements[row * C.width + col] = Cvalue;
            Cvalue = 0;
        }
    }

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

    MatMulSerial(A, B, C);

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
