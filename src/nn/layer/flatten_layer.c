#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer flatten_layer_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size)
{
    flatten_layer *fl = malloc(sizeof(flatten_layer));
    fl->x_rows = x_rows;
    fl->x_cols = x_cols;
    fl->x_depth = x_depth;
    fl->batch_size = batch_size;
    fl->y_size = x_rows * x_cols * x_depth;

    layer l;

    l.data = fl;

    l.forward = flatten_forward;
    l.backprop = flatten_backprop;
    l.destroy = flatten_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void flatten_forward(layer l, tens x, tens *y)
{
    flatten_layer *fl = (flatten_layer *)l.data;

    assert(x.type == TENS4D);
    assert(x.t4.rows == fl->x_rows);
    assert(x.t4.cols == fl->x_cols);
    assert(x.t4.depth == fl->x_depth);
    assert(x.t4.batches == fl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(fl->y_size, fl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->x_depth; ++j) {
            for (int k = 0; k < fl->x_rows; ++k) {
                for (int l = 0; l < fl->x_cols; ++l) {
                    int index = (j * fl->x_rows + k) * fl->x_cols + l;
                    mat_at(y->m, index, i) = tens4D_at(x.t4, k, l, j, i);
                }
            }
        }
    }
}

void flatten_backprop(layer l, tens dy, tens *dx, float rate)
{
    flatten_layer *fl = (flatten_layer *)l.data;

    assert(dy.type == MAT);
    assert(dy.m.rows == fl->y_size);
    assert(dy.m.cols == fl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(fl->x_rows, fl->x_cols,
                          fl->x_depth, fl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->x_depth; ++j) {
            for (int k = 0; k < fl->x_rows; ++k) {
                for (int l = 0; l < fl->x_cols; ++l) {
                    int index = (j * fl->x_rows + k) * fl->x_cols + l;
                    tens4D_at(dx->t4, k, l, j, i) = mat_at(dy.m, index, i);
                }
            }
        }
    }
}

void flatten_destroy(layer l)
{
    flatten_layer *fl = (flatten_layer *)l.data;

    free(fl);
}
