#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer concat_layer_alloc(int x_size, int batch_size, int steps)
{
    concat_layer *cl = malloc(sizeof(concat_layer));

    cl->x_size = x_size;
    cl->y_size = x_size * steps;
    cl->batch_size = batch_size;
    cl->steps = steps;

    layer l;

    l.type = CONCAT;
    l.data = cl;
    l.forward = concat_forward;
    l.backprop = concat_backprop;
    l.destroy = concat_destroy;
    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void concat_forward(layer l, void *x, void **y)
{
    concat_layer *cl = (concat_layer *)l.data;

    tens3D *tens3D_x = (tens3D *)x;

    assert(tens3D_x->rows == cl->x_size);
    assert(tens3D_x->cols == cl->batch_size);
    assert(tens3D_x->depth == cl->steps);

    mat *mat_y = malloc(sizeof(mat));
    *mat_y = mat_alloc(cl->y_size, cl->batch_size);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < cl->steps; ++i) {
        for (int j = 0; j < cl->x_size; ++j) {
            for (int k = 0; k < cl->batch_size; ++k) {
                mat_at(*mat_y, i * cl->x_size + j, k) = tens3D_at(*tens3D_x, j, k, i);
            }
        }
    }

    *y = mat_y;
}

void concat_backprop(layer l, void *dy, void **dx, float rate)
{
    concat_layer *cl = (concat_layer *)l.data;

    mat *mat_dy = (mat *)dy;

    assert(mat_dy->rows == cl->y_size);
    assert(mat_dy->cols == cl->batch_size);

    tens3D *tens3D_dx = malloc(sizeof(tens3D));
    *tens3D_dx = tens3D_alloc(cl->x_size, cl->batch_size, cl->steps);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < cl->steps; ++i) {
        for (int j = 0; j < cl->x_size; ++j) {
            for (int k = 0; k < cl->batch_size; ++k) {
                tens3D_at(*tens3D_dx, j, k, i) = mat_at(*mat_dy, i * cl->x_size + j, k);
            }
        }
    }

    *dx = tens3D_dx;
}

void concat_destroy(layer l)
{
    concat_layer *cl = (concat_layer *)l.data;

    free(cl);
}
