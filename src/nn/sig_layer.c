#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer sig_layer_2D_alloc(int x_size, int batch_size)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_rows = x_size;
    sl->x_cols = batch_size;
    sl->x_depth = 1;
    sl->x_batches = 1;

    sl->x_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_forward;
    l.backprop = sig_backprop;
    l.destroy = sig_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer sig_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->x_depth = batch_size;
    sl->x_batches = 1;

    sl->x_cache = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_forward;
    l.backprop = sig_backprop;
    l.destroy = sig_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer sig_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->x_depth = x_depth;
    sl->x_batches = batch_size;

    sl->x_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_forward;
    l.backprop = sig_backprop;
    l.destroy = sig_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void sig_forward(layer l, tens x, tens *y)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(x.rows == sl->x_rows);
    assert(x.cols == sl->x_cols);
    assert(x.depth == sl->x_depth);
    assert(x.batches == sl->x_batches);

    *y = tens4D_alloc(sl->x_rows, sl->x_cols,
                      sl->x_depth, sl->x_batches);

    tens_copy(sl->x_cache, x);

    tens_func(*y, x, sig);
}

void sig_backprop(layer l, tens dy, tens *dx, float rate)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(dy.rows == sl->x_rows);
    assert(dy.cols == sl->x_cols);
    assert(dy.depth == sl->x_depth);
    assert(dy.batches == sl->x_batches);

    *dx = tens4D_alloc(sl->x_rows, sl->x_cols,
                      sl->x_depth, sl->x_batches);

    tens_func(*dx, dy, dsig);
    tens_had(*dx, *dx, sl->x_cache);
}

void sig_destroy(layer l)
{
    sig_layer *sl = (sig_layer *)l.data;

    tens_destroy(sl->x_cache);

    free(sl);
}
