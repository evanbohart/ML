#include <stdlib.h>
#include <assert.h>
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

void tens4D_rand(tens4D t, double min, double max)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_rand(t.tens3Ds[i], min, max);
    }
}

void tens4D_normal(tens4D t, double mean, double stddev)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_normal(t.tens3Ds[i], mean, stddev);
    }
}

void tens4D_fill(tens4D t, double val)
{
    for (int i = 0; i < t.batches; ++i) {
        tens3D_fill(t.tens3Ds[i], val);
    }
}

void tens4D_copy(tens4D destination, tens4D t)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_copy(destination.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_sub(tens4D destination, tens4D t1, tens4D t2)
{
    assert(destination.batches == t1.batches);
    assert(destination.batches == t2.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_sub(destination.tens3Ds[i], t1.tens3Ds[i], t2.tens3Ds[i]);
    }
}

void tens4D_had(tens4D destination, tens4D t1, tens4D t2)
{
    assert(destination.batches == t1.batches);
    assert(destination.batches == t2.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_had(destination.tens3Ds[i], t1.tens3Ds[i], t2.tens3Ds[i]);
    }
}

void tens4D_trans(tens4D destination, tens4D t)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_trans(destination.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_180(tens4D destination, tens4D t)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_180(destination.tens3Ds[i], t.tens3Ds[i]);
    }
}

void tens4D_scale(tens4D destination, tens4D t, double a)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_scale(destination.tens3Ds[i], t.tens3Ds[i], a);
    }
}

void tens4D_func(tens4D destination, tens4D t, func f)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_func(destination.tens3Ds[i], t.tens3Ds[i], f);
    }
}

void tens4D_pad(tens4D destination, tens4D t, padding_t padding)
{
    assert(destination.batches == t.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_pad(destination.tens3Ds[i], t.tens3Ds[i], padding);
    }
}

void tens4D_maxpool(tens4D destination, tens4D t, tens4D mask, int pooling_size)
{
    assert(destination.batches == t.batches);
    assert(destination.batches == mask.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_maxpool(destination.tens3Ds[i], t.tens3Ds[i], mask.tens3Ds[i], pooling_size);
    }
}

void tens4D_maxunpool(tens4D destination, tens4D t, tens4D mask, int pooling_size)
{
    assert(destination.batches == t.batches);
    assert(destination.batches == mask.batches);

    for (int i = 0; i < destination.batches; ++i) {
        tens3D_maxunpool(destination.tens3Ds[i], t.tens3Ds[i], mask.tens3Ds[i], pooling_size);
    }
}

void tens4D_flatten(mat destination, tens4D t)
{
    assert(destination.rows == t.rows * t.cols * t.depth);
    assert(destination.cols == t.batches);

    int index = 0;

    for (int i = 0; i < t.depth; ++i) {
        for (int j = 0; j < t.rows; ++j) {
            for (int k = 0; k < t.cols; ++k) {
                for (int l = 0; l < t.batches; ++l) {
                    mat_at(destination, index, l) = tens4D_at(t, j, k, i, l);
                }

                ++index;
            }
        }
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
