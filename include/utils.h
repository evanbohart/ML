#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

double rand_double(double min, double max);
double rand_normal(double mean, double stddev);
void shuffle(void *arr, size_t type_size, int arr_size);
void get_path(char *path, char *file_name);
void clear_bin(char *path);

#ifdef __cplusplus
}
#endif

#endif
