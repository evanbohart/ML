#include <stdlib.h>
#include "nn.h"

layer tanh_layer_2D_alloc(int x_size, int batch_size)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->type = MAT;
    tl->x_rows = x_size;
    tl->batch_size = batch_size;

    tl->x_cache.type = MAT;
    tl->x_cache.m = mat_alloc(x_size, batch_size)

    layer l;

    l.type = TANH;
    l.data = tl;

    l.forward = tanh_2D_forward;
    l.backprop = tanh_2D_backprop;
    l.destroy = tanh_2D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer tanh_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->type = TENS3D;
    tl->x_rows = x_rows;
    tl->x_cols = x_cols;
    tl->batch_size = batch_size;

    tl->x_cache.type = TENS3D;
    tl->x_cache.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.type = TANH;
    l.data = tl;

    l.forward = tanh_3D_forward;
    l.backprop = tanh_3D_backprop;
    l.destroy = tanh_3D_destroy;

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

    tl->type = TENS4D;
    tl->x_rows = x_rows;
    tl->x_cols = x_cols;
    tl->x_depth = x_depth;
    tl->batch_size = batch_size;

    tl->x_cache.type = TENS4D;
    tl->x_cache.t4 = tens3D_alloc(x_rows, x_cols,
                                  x_depth, batch_size);

    layer l;

    l.type = TANH;
    l.data = tl;

    l.forward = tanh_4D_forward;
    l.backprop = tanh_4D_backprop;
    l.destroy = tanh_4D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void tanh_2D_forward(layer l, tens x, tens *y)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == MAT);

    assert(x.type == tl->type);
    assert(x.m.rows == tl->x_rows);
    assert(x.m.cols == tl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(tl->x_rows, tl->batch_size);

    mat_copy(tl->x_cache.m, x.m);

    mat_func(y->m, x.m, tanhf);
}

void tanh_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == MAT);

    assert(dy.type == tl->type);
    assert(dy.m.rows == tl->x_rows);
    assert(dy.m.cols == tl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(tl->x_rows, tl->batch_size);

    mat x_dtanh = mat_alloc(tl->x_rows, tl->batch_size);

    mat_func(x_dtanh, tl->x_cache.m, dtanh);
    mat_had(dx->m, dy.m, x_dtanh);

    free(x_dtanh.vals);
}

void tanh_2D_destroy(layer l)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == MAT);

    free(tl->x_cache.m.vals);

    free(tl);
}

void tanh_3D_forward(layer l, tens x, tens *y)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS3D);

    assert(x.type == tl->type);
    assert(x.t3.rows == tl->x_rows);
    assert(x.t3.cols == tl->x_cols);
    assert(x.t3.depth == tl->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(tl->x_rows, tl->x_cols, tl->batch_size);

    tens3D_copy(tl->x_cache.t3, x.t3);

    tens3D_func(y->t3, x.t3, tanhf);
}

void tanh_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS3D);

    assert(dy.type == tl->type);
    assert(dy.t3.rows == tl->x_rows);
    assert(dy.t3.cols == tl->x_cols);
    assert(dy.t3.depth == tl->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(tl->x_rows, tl->x_cols, tl->batch_size);

    tens3D x_dtanh = tens3D_alloc(tl->x_rows, tl->x_cols, tl->batch_size);

    tens3D_func(x_dtanh, tl->x_cache.t3, dtanh);
    tens3D_had(dx->t3, dy.t3, x_dtanh);

    tens3D_destroy(x_dtanh);
}

void tanh_3D_destroy(layer l)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS3D);

    tens3D_destroy(tl->x_cache.t3);

    free(tl);
}

void tanh_4D_forward(layer l, tens x, tens *y)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS4D);

    assert(x.type == tl->type);
    assert(x.t4.rows == tl->x_rows);
    assert(x.t4.cols == tl->x_cols);
    assert(x.t4.depth == tl->x_depth);
    assert(x.t4.batches == tl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(tl->x_rows, tl->x_cols,
                         tl->x_depth, tl->batch_size);

    tens4D_copy(tl->x_cache.t4, x.t4);

    tens4D_func(y->t4, x.t4, tanhf);
}

void tanh_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS4D);

    assert(dy.type == tl->type);
    assert(dy.t4.rows == tl->x_rows);
    assert(dy.t4.cols == tl->x_cols);
    assert(dy.t4.depth == tl->x_depth);
    assert(dy.t4.batches == tl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(tl->x_rows, tl->x_cols,
                          tl->x_depth, tl->batch_size);

    tens4D x_dtanh = tens4D_alloc(tl->x_rows, tl->x_cols,
                                  tl->x_depth, tl->batch_size);

    tens4D_func(x_dtanh, tl->x_cache.t4, dtanh);
    tens4D_had(dx->t4, dy.t4, x_dtanh);

    tens4D_destroy(x_dtanh);
}

void tanh_4D_destroy(layer l)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(tl->type == TENS4D);

    tens3D_destroy(tl->x_cache.t4);

    free(tl);
}
