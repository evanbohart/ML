#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#define get_directory _getcwd
#else
#include <unistd.h>
#define get_directory getcwd
#endif

float rand_float(float min, float max) { return (max - min) * rand() / RAND_MAX + min; }

float rand_normal(float mean, float stddev)
{
    float u1 = rand_float(0, 1);
    float u2 = rand_float(0, 1);

    while (u1 == 0) u1 = rand_float(0, 1);

    return stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) + mean;
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

void get_path(char *path, char *file_name)
{
    char directory[FILENAME_MAX];
    get_directory(directory, FILENAME_MAX);
    strcpy(path, directory);
    strcat(path, "/docs/");
    strcat(path, file_name);
}
