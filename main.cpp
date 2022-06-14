#include <stdio.h>
#include "mpi.h"
#include <math.h>


//////////////////// Example 1 ////////////////////
// Example Function 1: 1/((x -1)^2)
double f1(double x) {
    return 1/(pow((x-1), 2));
}

// Undefined integral of an Example Function 1: 1/(x -1)
double f1_undefined_integral(double x) {
    return 1/(1 - x);
}

//////////////////// Example 2 ////////////////////
// Example Function 2: (1 + (cos(x)^2)) / (1 + cos(2*x))
double f2(double x) {
    return (1 + pow(cos(x), 2))/(1 + cos(2*x));
}

// Undefined integral of an Example Function 1: 1/(x -1)
double f2_undefined_integral(double x) {
    return 0.5 * (x + tan(x));
}

//////////////// General Functions ////////////////
// Defined integral on an interval of any function of undefined integral
double f_defined_integral(double a, double b, double (*func)(double)) {
    return func(b) - func(a);
}

double absolute_error(double actual_value, double approximated_value) {
    return actual_value - approximated_value;
}

double midpoint_quadrature_integration(
        int world_size,
        int world_rank,
        int a,
        int b,
        double h,
        double (*func)(double)
        ) {
    /*
     * Calculates approximated integral by Midpoint Quadrature in a parallel manner
     *
     * Parameters:
     *     world_size: Number of Open MPI processes
     *     world_rank: ID of the current Open MPI process
     *     a: Beginning of the integration interval
     *     b: Ending of the integration interval
     *     h: Integration step size
     *     func: Function to be approximately integrated
     */

    /*
     * sum: Approximated integral on the given interval for the given function
     * local_sum: variable to store integral approximation on each sub-interval
     *                     calculated by a single process
     */
    double sum;
    double local_sum;

    /*
     * i: Variable for iterating through loop
     * n: Number integration sub-intervals which depends on step size
     * from: Beginning of a sub-interval on a specific iteration.
     * end: Ending of a sub-interval on a specific iteration.
     * midpoint: Is the middle point between "from" and "end" values.
     */
    int i;
    int n = (b - a) / h;
    double from, to, midpoint;

    for (i = world_rank; i < n; i += world_size) {
        from = a + i * h;
        to = a + (i + 1) * h;
        midpoint = (from + to) / 2;

        local_sum += h * func(midpoint);
    }

    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return sum;
}


int main(int argc, char *argv[])
{
    /*
    * world_size - number of processes
    * world_rank - the id of the process
    */
    int world_size, world_rank;

    // Initialize world environment and retrieving world size/rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /*
     * Integration parameters:
     * a: beginning of the integration interval
     * b: ending of the integration interval
     * h: integration step size
     */
    double h = 1e-8;
    int a = 3;
    int b = 4;

    // Choosing a function to approximately its integral and an undefined
    // integral of this function to calculate an error.
    double (*func)(double) = f2;
    double (*func_integral)(double) = f2_undefined_integral;

    // Calculating the approximated integral
    double approx_integral = midpoint_quadrature_integration(world_size, world_rank, a, b, h, func);

    // Showing the results
    if (world_rank == 0) {
        double actual_integral = f_defined_integral(a, b, func_integral);
        double abs_err = absolute_error(actual_integral, approx_integral);

        printf("Step: %g \n", h);
        printf("Actual: %.20g \n", actual_integral);
        printf("Approximated: %.20g\n", approx_integral);
        printf("Error: %g\n ", abs_err);
    }

    // Finish MPI environment
    MPI_Finalize();
}