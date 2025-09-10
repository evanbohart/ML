#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer relu_layer_2D_alloc(int x_size, int batch_size)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->x_rows = x_size;
    rl->x_cols = batch_size;
    rl->x_depth = 1;
    rl->x_batches = 1;

    rl->x_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = rl;

    l.forward = relu_forward;
    l.backprop = relu_backprop;
    l.destroy = relu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer relu_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->x_rows = x_rows;
    rl->x_cols = x_cols;
    rl->x_depth = batch_size;
    rl->x_batches = 1;

    rl->x_cache = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = rl;

    l.forward = relu_forward;
    l.backprop = relu_backprop;
    l.destroy = relu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer relu_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->x_rows = x_rows;
    rl->x_cols = x_cols;
    rl->x_depth = x_depth;
    rl->x_batches = batch_size;

    rl->x_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = rl;

    l.forward = relu_forward;
    l.backprop = relu_backprop;
    l.destroy = relu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void relu_forward(layer l, tens x, tens *y)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(x.rows == rl->x_rows);
    assert(x.cols == rl->x_cols);
    assert(x.depth == rl->x_depth);
    assert(x.batches == rl->x_batches);

    *y = tens4D_alloc(rl->x_rows, rl->x_cols,
                      rl->x_depth, rl->x_batches);

    tens_copy(rl->x_cache, x);

    tens_func(*y, x, relu);
}

void relu_backprop(layer l, tens dy, tens *dx, float rate)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(dy.rows == rl->x_rows);
    assert(dy.cols == rl->x_cols);
    assert(dy.depth == rl->x_depth);
    assert(dy.batches == rl->x_batches);

    *dx = tens4D_alloc(rl->x_rows, rl->x_cols,
                      rl->x_depth, rl->x_batches);

    tens_func(*dx, dy, drelu);
    tens_had(*dx, *dx, rl->x_cache);
}

void relu_destroy(layer l)
{
    relu_layer *rl = (relu_layer *)l.data;

    tens_destroy(rl->x_cache);

    free(rl);
}
