#include "nn.h"
#include <stdlib.h>
#include <assert.h>
#include <float.h>

tens tens_alloc(int rows, int cols, int depth)
{
	tens t;
	
	t.rows = rows;
	t.cols = cols;
	t.depth = depth;

	t.mats = malloc(depth * sizeof(mat));
    assert(t.mats);

	for (int i = 0; i < depth; ++i) {
		t.mats[i] = mat_alloc(rows, cols);
	}

	return t;
}

void tens_rand(tens t, double min, double max)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_rand(t.mats[i], min, max);
    }
}

void tens_normal(tens t, double mean, double stddev)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_normal(t.mats[i], mean, stddev);
    }
}

void tens_fill(tens t, double val)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_fill(t.mats[i], val);
    }
}

void tens_copy(tens destination, tens t)
{
    assert(destination.rows == t.rows);
    assert(destination.cols == t.cols);
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_copy(destination.mats[i], t.mats[i]);
    }
}

void tens_add(tens destination, tens t1, tens t2)
{
    assert(destination.rows == t1.rows);
    assert(destination.cols == t1.cols);
    assert(destination.depth == t1.depth);
    assert(t1.rows == t2.rows);
    assert(t1.cols == t2.cols);
    assert(t1.depth == t2.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_add(destination.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens_sub(tens destination, tens t1, tens t2)
{
    assert(destination.depth == t1.depth);
    assert(t1.depth == t2.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_sub(destination.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens_had(tens destination, tens t1, tens t2)
{
    assert(destination.depth == t1.depth);
    assert(t1.depth == t2.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_had(destination.mats[i], t1.mats[i], t2.mats[i]);
    }
}

void tens_scale(tens destination, tens t, double a)
{
    assert(destination.rows == t.rows);
    assert(destination.cols == t.cols);
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_scale(destination.mats[i], t.mats[i], a);
    }
}

void tens_func(tens destination, tens t, func f)
{
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_func(destination.mats[i], t.mats[i], f);
    }
}

void tens_pad(tens destination, tens t, padding_t padding)
{
    assert(destination.rows > t.rows);
    assert(destination.cols > t.cols);
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_pad(destination.mats[i], t.mats[i], padding);
    }
}

void tens_filter(tens destination, tens t, int row, int col)
{
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_filter(destination.mats[i], t.mats[i], row, col);
    }
}

void tens_maxpool(tens destination, tens mask, tens t, int pooling_size)
{
    assert(destination.depth == mask.depth);
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_maxpool(destination.mats[i], mask.mats[i], t.mats[i], pooling_size);
    }
}

void tens_maxunpool(tens destination, tens mask, tens t, int pooling_size)
{
    assert(destination.depth == mask.depth);
    assert(destination.depth == t.depth);

    for (int i = 0; i < destination.depth; ++i) {
        mat_maxunpool(destination.mats[i], mask.mats[i], t.mats[i], pooling_size);
    }
}

void tens_flatten(mat destination, tens t)
{
    assert(destination.rows == t.rows * t.cols * t.depth);
    assert(destination.cols == 1);

    int index = 0;

    for (int i = 0; i < t.depth; ++i) {
        for (int j = 0; j < t.rows; ++j) {
            for (int k = 0; k < t.cols; ++k) {
                mat_at(destination, index++, 0) = tens_at(t, j, k, i);
            }
        }
    }
}

void tens_print(tens t)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_print(t.mats[i]);
    }
}

void tens_destroy(tens *t)
{
    for (int i = 0; i < t->depth; ++i) {
        free(t->mats[i].vals);
    }

    free(t->mats);
}

void tens_load(tens *t, FILE *f)
{
    for (int i = 0; i < t->depth; ++i) {
        mat_load(&t->mats[i], f);
    }
}

void tens_save(tens t, FILE *f)
{
    for (int i = 0; i < t.depth; ++i) {
        mat_save(t.mats[i], f);
    }
}
