#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include "nn.h"

layer softmax_layer_alloc(int x_size, int batch_size)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_size = x_size;
    sl->batch_size = batch_size;

    sl->y_cache = mat_alloc(x_size, batch_size);

    layer l;

    l.data = sl;
    l.type = SOFTMAX;
    l.forward = softmax_forward;
    l.backprop = softmax_backprop;
    l.destroy = softmax_destroy;
    l.init = NULL;
    l.print = NULL;
    l.load = NULL;
    l.save = NULL;

    return l;
}

void softmax_forward(layer l, void *x, void **y)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    mat *mat_x = (mat *)x;

    assert(mat_x->rows == sl->x_size);
    assert(mat_x->cols == sl->batch_size);

    mat *mat_y = malloc(sizeof(mat));
    *mat_y = mat_alloc(sl->x_size, sl->batch_size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sl->batch_size; ++i) {
        float max = -FLT_MAX;

        for (int j = 0; j < sl->x_size; ++j) {
            if (mat_at(*mat_x, j, i) > max) max = mat_at(*mat_x, j, i);
        }

        float sum = 0.0f;

        for (int j = 0; j < sl->x_size; ++j) {
            float val = exp(mat_at(*mat_x, j, i) - max);

            mat_at(sl->y_cache, j, i) = val;

            sum += val;
        }

        for (int j = 0; j < sl->x_size; ++j) {
            mat_at(sl->y_cache, j, i) /= sum;
        }
    }

    mat_copy(*mat_y, sl->y_cache);

    *y = mat_y;
}

void softmax_backprop(layer l, void *dy, void **dx, float rate)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    mat *mat_dy = (mat *)dy;

    assert(mat_dy->rows == sl->x_size);
    assert(mat_dy->cols == sl->batch_size);

    mat *mat_dx = malloc(sizeof(mat));
    *mat_dx = mat_alloc(sl->x_size, sl->batch_size);

    for (int i = 0; i < sl->batch_size; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < sl->x_size; ++j) {
            sum += mat_at(*mat_dy, j, i) * mat_at(sl->y_cache, j, i);
        }

        for (int j = 0; j < sl->x_size; ++j) {
            mat_at(*mat_dx, j, i) = mat_at(sl->y_cache, j, i) * (mat_at(*mat_dy, j, i) - sum);
        }
    }

    *dx = mat_dx;
}

void softmax_destroy(layer l)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    free(sl->y_cache.vals);

    free(sl);
}
