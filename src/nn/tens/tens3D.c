#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <omp.h>
#include "nn.h"

tens3D tens3D_alloc(int rows, int cols, int depth)
{
	tens3D t;

	t.rows = rows;
	t.cols = cols;
	t.depth = depth;

	t.mats = malloc(depth * sizeof(mat));
    
	for (int i = 0; i < depth; ++i) {
		t.mats[i] = mat_alloc(rows, cols);
	}

	return t;
}

void tens3D_rand(tens3D t, float min, float max)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_rand(t.mats[i], min, max);
    }
}

void tens3D_normal(tens3D t, float mean, float stddev)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_normal(t.mats[i], mean, stddev);
    }
}

void tens3D_fill(tens3D t, float val)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < t.depth; ++i) {
        mat_fill(t.mats[i], val);
    }
}

void tens3D_copy(tens3D dest, tens3D t)
{
    assert(dest.rows == t.rows);
    assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_copy(dest.mats[i], t.mats[i]);
    }
}

void tens3D_add(tens3D dest, tens3D t1, tens3D t2)
{
    assert(dest.rows == t1.rows);
    assert(dest.cols == t1.cols);
    assert(dest.depth == t1.depth);
    assert(t1.rows == t2.rows);
    assert(t1.cols == t2.cols);
    assert(t1.depth == t2.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_add(dest.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens3D_sub(tens3D dest, tens3D t1, tens3D t2)
{
    assert(dest.depth == t1.depth);
    assert(t1.depth == t2.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_sub(dest.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens3D_had(tens3D dest, tens3D t1, tens3D t2)
{
    assert(dest.depth == t1.depth);
    assert(t1.depth == t2.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_had(dest.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens3D_trans(tens3D dest, tens3D t)
{
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_trans(dest.mats[i], t.mats[i]);
    }
}

void tens3D_180(tens3D dest, tens3D t)
{
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_180(dest.mats[i], t.mats[i]);
    }
}

void tens3D_scale(tens3D dest, tens3D t, float a)
{
    assert(dest.rows == t.rows);
    assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_scale(dest.mats[i], t.mats[i], a);
    }
}

void tens3D_func(tens3D dest, tens3D t, func f)
{
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_func(dest.mats[i], t.mats[i], f);
    }
}

void tens3D_pad(tens3D dest, tens3D t, padding_t padding)
{
    assert(dest.depth == t.depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dest.depth; ++i) {
        mat_pad(dest.mats[i], t.mats[i], padding);
    }
}

void tens3D_batchnorm(tens3D dest, tens3D t)
{
    assert(dest.rows == t.rows);
    assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
}

void tens3D_print(tens3D t)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_print(t.mats[i]);
    }
}

void tens3D_destroy(tens3D t)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < t.depth; ++i) {
        free(t.mats[i].vals);
    }

    free(t.mats);
}

void tens3D_save(tens3D t, FILE *f)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_save(t.mats[i], f);
    }
}

void tens3D_load(tens3D t, FILE *f)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_load(t.mats[i], f);
    }
}
