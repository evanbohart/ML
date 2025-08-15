#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

tens4D tens4D_alloc(int rows, int cols, int depth, int batches)
{
    tens4D t;

    t.rows = rows;
    t.cols = cols;
    t.depth = depth;
    t.batches = batches;

    t.tens3Ds = malloc(batches * sizeof(tens3D));

    for (int i = 0; i < batches; ++i) {
        t.tens3Ds[i] = tens3D_alloc(rows, cols, depth);
    }

    return t;
}

void tens4D_rand(tens4D t, float min, float max)
{
#pragma omp parallel for
    for (int i = 0; i < t.batches; ++i) {
        tens3D_rand(t.tens3Ds[i], min, max);
    }
}

void tens4D_normal(tens4D t, float mean, float stddev)
{
#pragma omp parallel for
    for (int i = 0; i < t.batches; ++i) {
        tens3D_normal(t.tens3Ds[i], mean, stddev);
    }
}

void tens4D_fill(tens4D t, float val)
{
    #pragma omp parallel for
    for (int i = 0; i < t.batches; ++i) {
        tens3D_fill(t.tens3Ds[i], val);
    }
}

void tens4D_copy(tens4D dest, tens4D t)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_copy(dest.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_add(tens4D dest, tens4D t1, tens4D t2)
{
    assert(dest.batches == t1.batches);
    assert(dest.batches == t2.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_add(dest.tens3Ds[i], t1.tens3Ds[i], t2.tens3Ds[i]);
    }
}

void tens4D_sub(tens4D dest, tens4D t1, tens4D t2)
{
    assert(dest.batches == t1.batches);
    assert(dest.batches == t2.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_sub(dest.tens3Ds[i], t1.tens3Ds[i], t2.tens3Ds[i]);
    }
}

void tens4D_had(tens4D dest, tens4D t1, tens4D t2)
{
    assert(dest.batches == t1.batches);
    assert(dest.batches == t2.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_had(dest.tens3Ds[i], t1.tens3Ds[i], t2.tens3Ds[i]);
    }
}

void tens4D_trans(tens4D dest, tens4D t)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_trans(dest.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_180(tens4D dest, tens4D t)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_180(dest.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_scale(tens4D dest, tens4D t, float a)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_scale(dest.tens3Ds[i], t.tens3Ds[i], a);
    }
}

void tens4D_func(tens4D dest, tens4D t, func f)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_func(dest.tens3Ds[i], t.tens3Ds[i], f);
    }
}

void tens4D_pad(tens4D dest, tens4D t, padding_t padding)
{
    assert(dest.batches == t.batches);

    #pragma omp parallel for
    for (int i = 0; i < dest.batches; ++i) {
        tens3D_pad(dest.tens3Ds[i], t.tens3Ds[i], padding);
    }
}

void tens4D_print(tens4D t)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_print(t.tens3Ds[i]);
    }
}

void tens4D_destroy(tens4D t)
{
    #pragma omp parallel for
    for (int i = 0; i < t.batches; ++i) {
        tens3D_destroy(t.tens3Ds[i]);
    }

    free(t.tens3Ds);
}

void tens4D_save(tens4D t, FILE *f)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_save(t.tens3Ds[i], f);
    }
}

void tens4D_load(tens4D t, FILE *f)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_load(t.tens3Ds[i], f);
    }
}
