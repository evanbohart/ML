#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer softmax_layer_2D_alloc(int x_size, int batch_size)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_rows = x_size;
    sl->x_cols = batch_size;
    sl->x_depth = 1;
    sl->x_batches = 1;

    sl->y_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = sl;

    l.forward = softmax_forward;
    l.backprop = softmax_backprop;
    l.destroy = softmax_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer softmax_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->x_depth = batch_size;
    sl->x_batches = 1;

    sl->y_cache = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = sl;

    l.forward = softmax_forward;
    l.backprop = softmax_backprop;
    l.destroy = softmax_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer softmax_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->x_depth = x_depth;
    sl->x_batches = batch_size;

    sl->y_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = sl;

    l.forward = softmax_forward;
    l.backprop = softmax_backprop;
    l.destroy = softmax_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void softmax_forward(layer l, tens x, tens *y)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    assert(x.rows == sl->x_rows);
    assert(x.cols == sl->x_cols);
    assert(x.depth == sl->x_depth);
    assert(x.batches == sl->x_batches);

    *y = tens4D_alloc(sl->x_rows, sl->x_cols, sl->x_depth, sl->x_batches);

    tens_softmax(*y, x);

    tens_copy(sl->y_cache, *y);
}

void softmax_backprop(layer l, tens dy, tens *dx, float rate)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    assert(dy.rows == sl->x_rows);
    assert(dy.cols == sl->x_cols);
    assert(dy.depth == sl->x_depth);
    assert(dy.batches == sl->x_batches);

    *dx = tens4D_alloc(sl->x_rows, sl->x_cols, sl->x_depth, sl->x_batches);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < sl->x_batches; ++i) {
        for (int j = 0; j < sl->x_depth; ++j) {
            for (int k = 0; k < sl->x_cols; ++k) {
                for (int l = 0; l < sl->x_rows; ++l) {
                    tens4D_at(*dx, l, k, j, i) = 0.0f;

                    float i_val = tens4D_at(sl->y_cache, l, k, j, i);

                    for (int m = 0; m < sl->x_rows; ++m) {
                        float j_val = tens4D_at(sl->y_cache, m, k, j, i);

                        if (l == m) {
                            tens4D_at(*dx, l, k, j, i) += i_val * (1.0f - j_val);
                        }
                        else {
                            tens4D_at(*dx, l, k, j, i) += -i_val * j_val;
                        }
                    }
                }
            }
        }
    }

    tens_had(*dx, *dx, dy);
}

void softmax_destroy(layer l)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    tens_destroy(sl->y_cache);

    free(sl);
}
