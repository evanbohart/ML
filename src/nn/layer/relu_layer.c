#include <stdlib.h>
#include "nn.h"

layer relu_layer_2D_alloc(int x_size, int batch_size)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->type = MAT;
    rl->x_rows = x_size;
    rl->batch_size = batch_size;

    rl->x_cache.type = MAT;
    rl->x_cache.m = mat_alloc(x_size, batch_size)

    layer l;

    l.type = RELU;
    l.data = rl;

    l.forward = relu_2D_forward;
    l.backprop = relu_2D_backprop;
    l.destroy = relu_2D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer relu_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->type = TENS3D;
    rl->x_rows = x_rows;
    rl->x_cols = x_cols;
    rl->batch_size = batch_size;

    rl->x_cache.type = TENS3D;
    rl->x_cache.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.type = RELU;
    l.data = rl;

    l.forward = relu_3D_forward;
    l.backprop = relu_3D_backprop;
    l.destroy = relu_3D_destroy;

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

    rl->type = TENS4D;
    rl->x_rows = x_rows;
    rl->x_cols = x_cols;
    rl->x_depth = x_depth;
    rl->batch_size = batch_size;

    rl->x_cache.type = TENS4D;
    rl->x_cache.t4 = tens3D_alloc(x_rows, x_cols,
                                  x_depth, batch_size);

    layer l;

    l.type = RELU;
    l.data = rl;

    l.forward = relu_4D_forward;
    l.backprop = relu_4D_backprop;
    l.destroy = relu_4D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void relu_2D_forward(layer l, tens x, tens *y)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == MAT);

    assert(x.type == rl->type);
    assert(x.m.rows == rl->x_rows);
    assert(x.m.cols == rl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(rl->x_rows, rl->batch_size);

    mat_copy(rl->x_cache.m, x.m);

    mat_func(y->m, x.m, relu);
}

void relu_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == MAT);

    assert(dy.type == rl->type);
    assert(dy.m.rows == rl->x_rows);
    assert(dy.m.cols == rl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(rl->x_rows, rl->batch_size);

    mat x_drelu = mat_alloc(rl->x_rows, rl->batch_size);

    mat_func(x_drelu, rl->x_cache.m, drelu);
    mat_had(dx->m, dy.m, x_drelu);

    free(x_drelu.vals);
}

void relu_2D_destroy(layer l)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == MAT);

    free(rl->x_cache.m.vals);

    free(rl);
}

void relu_3D_forward(layer l, tens x, tens *y)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS3D);

    assert(x.type == rl->type);
    assert(x.t3.rows == rl->x_rows);
    assert(x.t3.cols == rl->x_cols);
    assert(x.t3.depth == rl->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(rl->x_rows, rl->x_cols, rl->batch_size);

    tens3D_copy(rl->x_cache.t3, x.t3);

    tens3D_func(y->t3, x.t3, relu);
}

void relu_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS3D);

    assert(dy.type == rl->type);
    assert(dy.t3.rows == rl->x_rows);
    assert(dy.t3.cols == rl->x_cols);
    assert(dy.t3.depth == rl->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(rl->x_rows, rl->x_cols, rl->batch_size);

    tens3D x_drelu = tens3D_alloc(rl->x_rows, rl->x_cols, rl->batch_size);

    tens3D_func(x_drelu, rl->x_cache.t3, drelu);
    tens3D_had(dx->t3, dy.t3, x_drelu);

    tens3D_destroy(x_drelu);
}

void relu_3D_destroy(layer l)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS3D);

    tens3D_destroy(rl->x_cache.t3);

    free(rl);
}

void relu_4D_forward(layer l, tens x, tens *y)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS4D);

    assert(x.type == rl->type);
    assert(x.t4.rows == rl->x_rows);
    assert(x.t4.cols == rl->x_cols);
    assert(x.t4.depth == rl->x_depth);
    assert(x.t4.batches == rl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(rl->x_rows, rl->x_cols,
                         rl->x_depth, rl->batch_size);

    tens4D_copy(rl->x_cache.t4, x.t4);

    tens4D_func(y->t4, x.t4, relu);
}

void relu_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS4D);

    assert(dy.type == rl->type);
    assert(dy.t4.rows == rl->x_rows);
    assert(dy.t4.cols == rl->x_cols);
    assert(dy.t4.depth == rl->x_depth);
    assert(dy.t4.batches == rl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(rl->x_rows, rl->x_cols,
                          rl->x_depth, rl->batch_size);

    tens4D x_drelu = tens4D_alloc(rl->x_rows, rl->x_cols,
                                  rl->x_depth, rl->batch_size);

    tens4D_func(x_drelu, rl->x_cache.t4, drelu);
    tens4D_had(dx->t4, dy.t4, x_drelu);

    tens4D_destroy(x_drelu);
}

void relu_4D_destroy(layer l)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(rl->type == TENS4D);

    tens3D_destroy(rl->x_cache.t4);

    free(rl);
}
