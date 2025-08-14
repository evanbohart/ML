#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer softmax_layer_alloc(int x_size, int batch_size)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_size = x_size;
    sl->batch_size = batch_size;

    sl->y_cache = mat_alloc(x_size, batch_size);

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

    assert(x.type == MAT);
    assert(x.m.rows == sl->x_size);
    assert(x.m.cols == sl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(sl->x_size, sl->batch_size);

    mat_softmax(y->m, x.m);
    mat_copy(sl->y_cache, y->m);
}

void softmax_backprop(layer l, tens dy, tens *dx, float rate)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    assert(dy.type == MAT);
    assert(dy.m.rows == sl->x_size);
    assert(dy.m.cols == sl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(sl->x_size, sl->batch_size);

    tens3D y_diag = tens3D_alloc(sl->x_size, sl->x_size, sl->batch_size);
    tens3D_fill(y_diag, 0.0f);

    tens3D y = tens3D_alloc(sl->x_size, 1, sl->batch_size);
    tens3D y_T = tens3D_alloc(1, sl->x_size, sl->batch_size);
    tens3D y_y_T = tens3D_alloc(sl->x_size, sl->x_size, sl->batch_size);
    tens3D J = tens3D_alloc(sl->x_size, sl->x_size, sl->batch_size);;
    tens3D dy_b = tens3D_alloc(sl->x_size, 1, sl->batch_size);
    tens3D dx_b = tens3D_alloc(sl->x_size, 1, sl->batch_size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sl->batch_size; ++i) {
        for (int j = 0; j < sl->x_size; ++j) {
            tens3D_at(y_diag, j, j, i) = mat_at(sl->y_cache, j, i);
            tens3D_at(y, j, 0, i) = mat_at(sl->y_cache, j, i);
            tens3D_at(y_T, 0, j, i) = mat_at(sl->y_cache, j, i);
            tens3D_at(dy_b, j, 0, i) = mat_at(dy.m, j, i);
        }

        mat_dot(y_y_T.mats[i], y.mats[i], y_T.mats[i]);
        mat_sub(J.mats[i], y_diag.mats[i], y_y_T.mats[i]);
        mat_dot(dx_b.mats[i], J.mats[i], dy_b.mats[i]);

        for (int j = 0; j < sl->x_size; ++j) {
            mat_at(dx->m, j, i) = tens3D_at(dx_b, j, 0, i);
        }
    }

    tens3D_destroy(y);
    tens3D_destroy(y_T);
    tens3D_destroy(y_y_T);
    tens3D_destroy(J);
    tens3D_destroy(dy_b);
    tens3D_destroy(dx_b);
}

void softmax_destroy(layer l)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    free(sl->y_cache.vals);

    free(sl);
}
