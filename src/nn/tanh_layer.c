#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

layer tanh_layer_2D_alloc(int x_size, int batch_size)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->x_rows = x_size;
    tl->x_cols = batch_size;
    tl->x_depth = 1;
    tl->x_batches = 1;

    tl->x_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = tl;

    l.forward = tanh_forward;
    l.backprop = tanh_backprop;
    l.destroy = tanh_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer tanh_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->x_rows = x_rows;
    tl->x_cols = x_cols;
    tl->x_depth = batch_size;
    tl->x_batches = 1;

    tl->x_cache = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = tl;

    l.forward = tanh_forward;
    l.backprop = tanh_backprop;
    l.destroy = tanh_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer tanh_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->x_rows = x_rows;
    tl->x_cols = x_cols;
    tl->x_depth = x_depth;
    tl->x_batches = batch_size;

    tl->x_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = tl;

    l.forward = tanh_forward;
    l.backprop = tanh_backprop;
    l.destroy = tanh_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void tanh_forward(layer l, tens x, tens *y)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(x.rows == tl->x_rows);
    assert(x.cols == tl->x_cols);
    assert(x.depth == tl->x_depth);
    assert(x.batches == tl->x_batches);

    *y = tens4D_alloc(tl->x_rows, tl->x_cols,
                      tl->x_depth, tl->x_batches);

    tens_copy(tl->x_cache, x);

    tens_func(*y, x, tanhf);
}

void tanh_backprop(layer l, tens dy, tens *dx, float rate)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(dy.rows == tl->x_rows);
    assert(dy.cols == tl->x_cols);
    assert(dy.depth == tl->x_depth);
    assert(dy.batches == tl->x_batches);

    *dx = tens4D_alloc(tl->x_rows, tl->x_cols,
                      tl->x_depth, tl->x_batches);

    tens_func(*dx, dy, dtanh);
    tens_had(*dx, *dx, tl->x_cache);
}

void tanh_destroy(layer l)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    tens_destroy(tl->x_cache);

    free(tl);
}
