#include "utils.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#ifdef WINDOWS
#include <direct.h>
#define get_directory _getcwd
#else
#include <unistd.h>
#define get_directory getcwd
#endif

double rand_double(double min, double max) { return (max - min) * rand() / RAND_MAX + min; }

double rand_normal(double mean, double stddev)
{
    static double n2 = 0;
    static bool cached = false;

    if (!cached) {
        double x;
        double y;
        double r;
        do {
            x = rand_double(-1, 1);
            y = rand_double(-1, 1);
            r = x * x + y * y;
        }
        while (r == 0 || r > 1);

        double n1 = x * sqrt(-2.0 * log(r) / r);
        n2 = y * n1;
        cached = true;
        return n1 * stddev + mean;
    }
    else {
        cached = false;
        return n2 * stddev + mean;
    }
}

void shuffle(void *arr, size_t type_size, int arr_size)
{
    char *temp_arr = (char *) arr;
	
    for (int i = arr_size - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        for (int k = 0; k < (int) type_size; ++k) {
            char temp = temp_arr[i * type_size + k];
            temp_arr[i * type_size + k] = temp_arr[j * type_size + k];
            temp_arr[j * type_size + k] = temp;
        }
    }
}

void get_path(char *path, char *file_name) {
    char directory[FILENAME_MAX];
    get_directory(directory, FILENAME_MAX);
    strcpy(path, directory);
    strcat(path, "\\docs\\");
    strcat(path, file_name);
}

void clear_bin(char *path)
{
    FILE *f = fopen(path, "wb");
    fclose(f);
}
