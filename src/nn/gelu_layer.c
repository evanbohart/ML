#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer gelu_layer_2D_alloc(int x_size, int batch_size)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->x_rows = x_size;
    gl->x_cols = batch_size;
    gl->x_depth = 1;
    gl->x_batches = 1;

    gl->x_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = gl;

    l.forward = gelu_forward;
    l.backprop = gelu_backprop;
    l.destroy = gelu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer gelu_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->x_rows = x_rows;
    gl->x_cols = x_cols;
    gl->x_depth = batch_size;
    gl->x_batches = 1;

    gl->x_cache = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = gl;

    l.forward = gelu_forward;
    l.backprop = gelu_backprop;
    l.destroy = gelu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer gelu_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->x_rows = x_rows;
    gl->x_cols = x_cols;
    gl->x_depth = x_depth;
    gl->x_batches = batch_size;

    gl->x_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = gl;

    l.forward = gelu_forward;
    l.backprop = gelu_backprop;
    l.destroy = gelu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void gelu_forward(layer l, tens x, tens *y)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(x.rows == gl->x_rows);
    assert(x.cols == gl->x_cols);
    assert(x.depth == gl->x_depth);
    assert(x.batches == gl->x_batches);

    *y = tens4D_alloc(gl->x_rows, gl->x_cols,
                      gl->x_depth, gl->x_batches);

    tens_copy(gl->x_cache, x);

    tens_func(*y, x, gelu);
}

void gelu_backprop(layer l, tens dy, tens *dx, float rate)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(dy.rows == gl->x_rows);
    assert(dy.cols == gl->x_cols);
    assert(dy.depth == gl->x_depth);
    assert(dy.batches == gl->x_batches);

    *dx = tens4D_alloc(gl->x_rows, gl->x_cols,
                      gl->x_depth, gl->x_batches);

    tens_func(*dx, dy, dgelu);
    tens_had(*dx, *dx, gl->x_cache);
}

void gelu_destroy(layer l)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    tens_destroy(gl->x_cache);

    free(gl);
}
