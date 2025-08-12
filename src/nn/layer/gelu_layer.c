#include <stdlib.h>
#include "nn.h"

layer gelu_layer_2D_alloc(int x_size, int batch_size)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->type = MAT;
    gl->x_rows = x_size;
    gl->batch_size = batch_size;

    gl->x_cache.type = MAT;
    gl->x_cache.m = mat_alloc(x_size, batch_size)

    layer l;

    l.type = GELU;
    l.data = gl;

    l.forward = gelu_2D_forward;
    l.backprop = gelu_2D_backprop;
    l.destroy = gelu_2D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer gelu_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->type = TENS3D;
    gl->x_rows = x_rows;
    gl->x_cols = x_cols;
    gl->batch_size = batch_size;

    gl->x_cache.type = TENS3D;
    gl->x_cache.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.type = GELU;
    l.data = gl;

    l.forward = gelu_3D_forward;
    l.backprop = gelu_3D_backprop;
    l.destroy = gelu_3D_destroy;

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

    gl->type = TENS4D;
    gl->x_rows = x_rows;
    gl->x_cols = x_cols;
    gl->x_depth = x_depth;
    gl->batch_size = batch_size;

    gl->x_cache.type = TENS4D;
    gl->x_cache.t4 = tens3D_alloc(x_rows, x_cols,
                                  x_depth, batch_size);

    layer l;

    l.type = GELU;
    l.data = gl;

    l.forward = gelu_4D_forward;
    l.backprop = gelu_4D_backprop;
    l.destroy = gelu_4D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void gelu_2D_forward(layer l, tens x, tens *y)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == MAT);

    assert(x.type == gl->type);
    assert(x.m.rows == gl->x_rows);
    assert(x.m.cols == gl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(gl->x_rows, gl->batch_size);

    mat_copy(gl->x_cache.m, x.m);

    mat_func(y->m, x.m, gelu);
}

void gelu_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == MAT);

    assert(dy.type == gl->type);
    assert(dy.m.rows == gl->x_rows);
    assert(dy.m.cols == gl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(gl->x_rows, gl->batch_size);

    mat x_dgelu = mat_alloc(gl->x_rows, gl->batch_size);

    mat_func(x_dgelu, gl->x_cache.m, dgelu);
    mat_had(dx->m, dy.m, x_dgelu);

    free(x_dgelu.vals);
}

void gelu_2D_destroy(layer l)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == MAT);

    free(gl->x_cache.m.vals);

    free(gl);
}

void gelu_3D_forward(layer l, tens x, tens *y)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS3D);

    assert(x.type == gl->type);
    assert(x.t3.rows == gl->x_rows);
    assert(x.t3.cols == gl->x_cols);
    assert(x.t3.depth == gl->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(gl->x_rows, gl->x_cols, gl->batch_size);

    tens3D_copy(gl->x_cache.t3, x.t3);

    tens3D_func(y->t3, x.t3, gelu);
}

void gelu_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS3D);

    assert(dy.type == gl->type);
    assert(dy.t3.rows == gl->x_rows);
    assert(dy.t3.cols == gl->x_cols);
    assert(dy.t3.depth == gl->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(gl->x_rows, gl->x_cols, gl->batch_size);

    tens3D x_dgelu = tens3D_alloc(gl->x_rows, gl->x_cols, gl->batch_size);

    tens3D_func(x_dgelu, gl->x_cache.t3, dgelu);
    tens3D_had(dx->t3, dy.t3, x_dgelu);

    tens3D_destroy(x_dgelu);
}

void gelu_3D_destroy(layer l)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS3D);

    tens3D_destroy(gl->x_cache.t3);

    free(gl);
}

void gelu_4D_forward(layer l, tens x, tens *y)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS4D);

    assert(x.type == gl->type);
    assert(x.t4.rows == gl->x_rows);
    assert(x.t4.cols == gl->x_cols);
    assert(x.t4.depth == gl->x_depth);
    assert(x.t4.batches == gl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(gl->x_rows, gl->x_cols,
                         gl->x_depth, gl->batch_size);

    tens4D_copy(gl->x_cache.t4, x.t4);

    tens4D_func(y->t4, x.t4, gelu);
}

void gelu_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS4D);

    assert(dy.type == gl->type);
    assert(dy.t4.rows == gl->x_rows);
    assert(dy.t4.cols == gl->x_cols);
    assert(dy.t4.depth == gl->x_depth);
    assert(dy.t4.batches == gl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(gl->x_rows, gl->x_cols,
                          gl->x_depth, gl->batch_size);

    tens4D x_dgelu = tens4D_alloc(gl->x_rows, gl->x_cols,
                                  gl->x_depth, gl->batch_size);

    tens4D_func(x_dgelu, gl->x_cache.t4, dgelu);
    tens4D_had(dx->t4, dy.t4, x_dgelu);

    tens4D_destroy(x_dgelu);
}

void gelu_4D_destroy(layer l)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(gl->type == TENS4D);

    tens3D_destroy(gl->x_cache.t4);

    free(gl);
}
