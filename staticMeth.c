#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

// Define the rank of the master process
#define MASTER_RANK 0

// This function calculates the Mandelbrot set for a given section of the image
void calculate_mandelbrot(int width, int height, int start_row, int end_row, double left, double right, double lower, double upper, int* output) {
    // Set the maximum number of iterations to determine if a point is in the Mandelbrot set
    int max_iterations = 1000;
    double x, y, x_new, y_new, x_squared, y_squared;
    int iteration;

    // Loop over all pixels in the section of the image specified by start_row and end_row
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            // Calculate the coordinates of the current pixel
            x = left + (right - left) * j / width;
            y = lower + (upper - lower) * i / height;

            // Initialize the iteration count and complex coordinates
            x_new = 0.0;
            y_new = 0.0;
            x_squared = 0.0;
            y_squared = 0.0;
            iteration = 0;

            // Iterate the complex function until the maximum number of iterations is reached or the point is outside the Mandelbrot set
            while (x_squared + y_squared < 4.0 && iteration < max_iterations) {
                y_new = 2 * x_new * y + y;
                x_new = x_squared - y_squared + x;
                x_squared = x_new * x_new;
                y_squared = y_new * y_new;
                iteration++;
            }

            // Save the iteration count for the current pixel
            output[i * width + j] = iteration;
        }
    }
}

// The main function
int main(int argc, char** argv) {
    // Start timing the program
    clock_t start = clock();

    // Set the dimensions and zoom level of the image
    int width = 800, height = 800;
    double left = -2.0, right = 1.0, lower = -1.5, upper = 1.5;
    int max_iterations = 1000;

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the image into sections for each MPI process to calculate
    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;
    if (rank == size - 1) {
        end_row = height;
    }
    int row_count = end_row - start_row;

    // Allocate memory for the local output buffer
    int* local_output = (int*) malloc(row_count * width * sizeof(int));

    // Calculate the Mandelbrot set for the local section of the image
    calculate_mandelbrot(width, height, start_row, end_row, left, right, lower, upper, local_output);

    // Allocate memory for the global output buffer on the master process
    int* global_output = NULL;
    if (rank == MASTER_RANK) {
        global_output = (int*) malloc(width * height * sizeof(int));
    }


    
    MPI_Gather(local_output, row_count * width, MPI_INT, global_output, row_count * width, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

// If the current process is the master process
if (rank == MASTER_RANK) {
    // Open a file to write the output image in PGM format
    FILE* fp = fopen("mandelbrot.pgm", "wb");
    // Write the PGM header to the file, which includes the image dimensions and maximum iterations
    fprintf(fp, "P2\n%d %d\n%d\n", width, height, max_iterations - 1);
    // Write the pixel values to the file
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(fp, "%d ", global_output[i * width + j]);
        }
        fprintf(fp, "\n");
    }
    // Close the file and free the memory allocated for the global output array
    fclose(fp);
    free(global_output);
}

// Free the memory allocated for the local output array
free(local_output);

// Finalize the MPI library
MPI_Finalize();

// Compute the CPU time used and print it to the console
end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf("CPU time used: %f seconds\n", cpu_time_used);
    return 0;
}