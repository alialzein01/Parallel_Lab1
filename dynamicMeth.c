#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// Define the width and height of the image and maximum number of iterations
#define WIDTH 800
#define HEIGHT 600
#define MAX_ITER 1000

int main(int argc, char** argv) {
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    
    // Initialize MPI environment and get rank and size of the process
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Start a timer to measure execution time in MPI environment
    double t_start = MPI_Wtime();

    // Calculate the number of tasks and tasks per process
    int num_tasks = WIDTH * HEIGHT;
    int tasks_per_proc = ceil(num_tasks / (double)size);

    // Calculate the start and end task indices for this process
    int start_task = rank * tasks_per_proc;
    int end_task = fmin(start_task + tasks_per_proc, num_tasks);

    // Initialize the Mandelbrot set array to all 0s
    int mandelbrot[HEIGHT][WIDTH] = {0};

    // Compute the Mandelbrot set for each task assigned to this process
    for (int task = start_task; task < end_task; task++) {
        // Convert the task index to coordinates in the image
        int x = task % WIDTH;
        int y = task / WIDTH;

        // Calculate the corresponding complex number c for the given pixel coordinates
        double c_re = (x - WIDTH/2.0)*4.0/WIDTH;
        double c_im = (y - HEIGHT/2.0)*4.0/WIDTH;

        // Initialize the real and imaginary parts of z to 0
        double z_re = 0, z_im = 0;

        // Iterate the Mandelbrot equation until the sequence diverges or maximum iterations reached
        int iter;
        for (iter = 0; iter < MAX_ITER; iter++) {
            double z_re_new = z_re*z_re - z_im*z_im + c_re;
            double z_im_new = 2*z_re*z_im + c_im;
            z_re = z_re_new;
            z_im = z_im_new;

            if (z_re*z_re + z_im*z_im > 4) {
                break;
            }
        }
        // Store the number of iterations it took to diverge (or MAX_ITER if sequence didn't diverge) in the Mandelbrot set array
        mandelbrot[y][x] = iter;
    }

    // Gather the computed Mandelbrot set values from all processes to the root process
    int* recvcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        // Calculate the number of tasks and starting indices for each process
        int num_tasks_i = i == size - 1 ? num_tasks - i * tasks_per_proc : tasks_per_proc;
        recvcounts[i] = num_tasks_i;
        displs[i] = i * tasks_per_proc;
    }

    int* mandelbrot_all = NULL;
    if (rank == 0) {
        // Allocate memory to store all the Mandelbrot set values
        mandelbrot_all = malloc(num_tasks * sizeof(int));
    }
    MPI_Gatherv(&(mandelbrot[0][start_task]), end_task - start_task, MPI_INT,
                mandelbrot_all, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* fp = fopen("mandelbrot.ppm", "wb");
        fprintf(fp, "P6 %d %d 255\n", WIDTH, HEIGHT);
        for (int i = 0; i < num_tasks; i++) {
            int color = mandelbrot_all[i] * 255 / MAX_ITER;
            fputc(color, fp);
            fputc(color, fp);
            fputc(color, fp);
        }
        fclose(fp);
        free(mandelbrot_all);

        double t_end = MPI_Wtime();
        printf("Execution time: %.2f seconds\n", t_end - t_start);
    }

    MPI_Finalize();
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    
    return 0;
}