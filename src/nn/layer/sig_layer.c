#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer sig_layer_2D_alloc(int x_size, int batch_size)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_type = MAT;
    sl->x_rows = x_size;
    sl->batch_size = batch_size;

    sl->x_cache.type = MAT;
    sl->x_cache.m = mat_alloc(x_size, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_2D_forward;
    l.backprop = sig_2D_backprop;
    l.destroy = sig_2D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer sig_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_type = TENS3D;
    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->batch_size = batch_size;

    sl->x_cache.type = TENS3D;
    sl->x_cache.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_3D_forward;
    l.backprop = sig_3D_backprop;
    l.destroy = sig_3D_destroy;

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

    sl->x_type = TENS4D;
    sl->x_rows = x_rows;
    sl->x_cols = x_cols;
    sl->x_depth = x_depth;
    sl->batch_size = batch_size;

    sl->x_cache.type = TENS4D;
    sl->x_cache.t4 = tens4D_alloc(x_rows, x_cols,
                                  x_depth, batch_size);

    layer l;

    l.data = sl;

    l.forward = sig_4D_forward;
    l.backprop = sig_4D_backprop;
    l.destroy = sig_4D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void sig_2D_forward(layer l, tens x, tens *y)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == MAT);

    assert(x.type == MAT);
    assert(x.m.rows == sl->x_rows);
    assert(x.m.cols == sl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(sl->x_rows, sl->batch_size);

    mat_copy(sl->x_cache.m, x.m);

    mat_func(y->m, x.m, sig);
}

void sig_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == MAT);

    assert(dy.type == MAT);
    assert(dy.m.rows == sl->x_rows);
    assert(dy.m.cols == sl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(sl->x_rows, sl->batch_size);

    mat x_dsig = mat_alloc(sl->x_rows, sl->batch_size);

    mat_func(x_dsig, sl->x_cache.m, dsig);
    mat_had(dx->m, dy.m, x_dsig);

    free(x_dsig.vals);
}

void sig_2D_destroy(layer l)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == MAT);

    free(sl->x_cache.m.vals);

    free(sl);
}

void sig_3D_forward(layer l, tens x, tens *y)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS3D);

    assert(x.type == TENS3D);
    assert(x.t3.rows == sl->x_rows);
    assert(x.t3.cols == sl->x_cols);
    assert(x.t3.depth == sl->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(sl->x_rows, sl->x_cols, sl->batch_size);

    tens3D_copy(sl->x_cache.t3, x.t3);

    tens3D_func(y->t3, x.t3, sig);
}

void sig_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS3D);

    assert(dy.type == TENS3D);
    assert(dy.t3.rows == sl->x_rows);
    assert(dy.t3.cols == sl->x_cols);
    assert(dy.t3.depth == sl->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(sl->x_rows, sl->x_cols, sl->batch_size);

    tens3D x_dsig = tens3D_alloc(sl->x_rows, sl->x_cols, sl->batch_size);

    tens3D_func(x_dsig, sl->x_cache.t3, dsig);
    tens3D_had(dx->t3, dy.t3, x_dsig);

    tens3D_destroy(x_dsig);
}

void sig_3D_destroy(layer l)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS3D);

    tens3D_destroy(sl->x_cache.t3);

    free(sl);
}

void sig_4D_forward(layer l, tens x, tens *y)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS4D);

    assert(x.type == TENS4D);
    assert(x.t4.rows == sl->x_rows);
    assert(x.t4.cols == sl->x_cols);
    assert(x.t4.depth == sl->x_depth);
    assert(x.t4.batches == sl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(sl->x_rows, sl->x_cols,
                         sl->x_depth, sl->batch_size);

    tens4D_copy(sl->x_cache.t4, x.t4);

    tens4D_func(y->t4, x.t4, sig);
}

void sig_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS4D);

    assert(dy.type == TENS4D);
    assert(dy.t4.rows == sl->x_rows);
    assert(dy.t4.cols == sl->x_cols);
    assert(dy.t4.depth == sl->x_depth);
    assert(dy.t4.batches == sl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(sl->x_rows, sl->x_cols,
                          sl->x_depth, sl->batch_size);

    tens4D x_dsig = tens4D_alloc(sl->x_rows, sl->x_cols,
                                 sl->x_depth, sl->batch_size);

    tens4D_func(x_dsig, sl->x_cache.t4, dsig);
    tens4D_had(dx->t4, dy.t4, x_dsig);

    tens4D_destroy(x_dsig);
}

void sig_4D_destroy(layer l)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(sl->x_type == TENS4D);

    tens4D_destroy(sl->x_cache.t4);

    free(sl);
}
