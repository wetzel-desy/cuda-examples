// This example is taken from https://devblogs.nvidia.com/even-easier-introduction-cuda/
#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// get_walltime function for time measurement
double get_walltime_(double* wcTime) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
  return 0.0;
}

void get_walltime(double* wcTime) {
  get_walltime_(wcTime);
}

// function to add the elements of two arrays
 void add(int n, float *x, float *y)
 {
   for (int i = 0; i < n; i++)
         y[i] = x[i] + y[i];
 }

int main(void)
{
    int N = 1<<20; // 1M elements

    float *x = new float[N];
    float *y = new float[N];
    double delta, finish, start;
    double flops, nd;

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // start time measurement
    get_walltime(&start);

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    // stop time measurement
    get_walltime(&finish);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // calculating time delta and Mflops
    delta = (finish - start);
    nd = (double) N;
    flops = nd/(delta * 1000000.);

    std::cout << ">>>>> finish: " << finish << std::endl;
    std::cout << ">>>>> delta: " << delta << std::endl;
    std::cout << ">>>>> Mflops: " << flops << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}
