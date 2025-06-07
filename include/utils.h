#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

float rand_float(float min, float max);
float rand_normal(float mean, float stddev);
void shuffle(void *arr, size_t type_size, int arr_size);
void get_path(char *path, char *file_name);

#ifdef __cplusplus
}
#endif

#endif
