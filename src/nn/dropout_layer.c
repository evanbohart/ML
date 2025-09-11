#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

layer dropout_layer_2D_alloc(int x_rows, int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_rows = x_rows;
    dl->x_cols = batch_size;
    dl->x_depth = 1;
    dl->x_batches = 1;

    dl->rate = rate;

    dl->mask = tens2D_alloc(x_rows, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_forward;
    l.backprop = dropout_backprop;
    l.destroy = dropout_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer dropout_layer_3D_alloc(int x_rows, int x_cols,
                             int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_rows = x_rows;
    dl->x_cols = x_cols;
    dl->x_depth = batch_size;
    dl->x_batches = 1;

    dl->rate = rate;

    dl->mask = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_forward;
    l.backprop = dropout_backprop;
    l.destroy = dropout_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer dropout_layer_4D_alloc(int x_rows, int x_cols,
                             int x_depth, int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_rows = x_rows;
    dl->x_cols = x_cols;
    dl->x_depth = x_depth;
    dl->x_batches = batch_size;

    dl->rate = rate;

    dl->mask = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_forward;
    l.backprop = dropout_backprop;
    l.destroy = dropout_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void dropout_forward(layer l, tens x, tens *y)
{
    dropout_layer *dl = (dropout_layer *)l.data;


    assert(x.rows == dl->x_rows);
    assert(x.cols == dl->x_cols);
    assert(x.depth == dl->x_depth);
    assert(x.batches == dl->x_batches);

    *y = tens4D_alloc(dl->x_rows, dl->x_cols, dl->x_depth, dl->x_batches);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->x_batches; ++i) {
        for (int j = 0; j < dl->x_depth; ++j) {
            for (int k = 0; k < dl->x_rows; ++k) {
                for (int l = 0; l < dl->x_cols; ++l) {
                    tens4D_at(dl->mask, k, l, j, i) =
                        rand_float(0.0f, 1.0f) > dl->rate ? 1.0f : 0.0f;
                }
            }
        }
    }

    tens_had(*y, x, dl->mask);
}

void dropout_backprop(layer l, tens dy, tens *dx, float rate)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dy.rows == dl->x_rows);
    assert(dy.cols == dl->x_cols);
    assert(dy.depth == dl->x_depth);
    assert(dy.batches == dl->x_batches);

    *dx = tens4D_alloc(dl->x_rows, dl->x_cols,
                       dl->x_depth, dl->x_batches);

    tens_had(*dx, dy, dl->mask);
}

void dropout_destroy(layer l)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    tens_destroy(dl->mask);

    free(dl);
}
