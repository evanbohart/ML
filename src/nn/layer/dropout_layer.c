#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include "utils.h"

layer dropout_layer_2D_alloc(int x_rows, int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_type = MAT;
    dl->x_rows = x_rows;
    dl->batch_size = batch_size;
    dl->rate = rate;

    dl->mask.type = MAT;
    dl->mask.m = mat_alloc(x_rows, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_2D_forward;
    l.backprop = dropout_2D_backprop;
    l.destroy = dropout_2D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer dropout_layer_3D_alloc(int x_rows, int x_cols,
                             int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_type = TENS3D;
    dl->x_rows = x_rows;
    dl->x_cols = x_cols;
    dl->batch_size = batch_size;
    dl->rate = rate;

    dl->mask.type = TENS3D;
    dl->mask.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_3D_forward;
    l.backprop = dropout_3D_backprop;
    l.destroy = dropout_3D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

layer dropout_layer_4D_alloc(int x_rows, int x_cols,
                             int x_depth, int batch_size, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_type = TENS4D;
    dl->x_rows = x_rows;
    dl->x_cols = x_cols;
    dl->x_depth = x_depth;
    dl->batch_size = batch_size;
    dl->rate = rate;

    dl->mask.type = TENS4D;
    dl->mask.t4 = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    layer l;

    l.data = dl;

    l.forward = dropout_4D_forward;
    l.backprop = dropout_4D_backprop;
    l.destroy = dropout_4D_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void dropout_2D_forward(layer l, tens x, tens *y)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == MAT);

    assert(x.type == MAT);
    assert(x.m.rows == dl->x_rows);
    assert(x.m.cols == dl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(dl->x_rows, dl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->x_rows; ++i) {
        for (int j = 0; j < dl->batch_size; ++j) {
            mat_at(dl->mask.m, i, j) =
                rand_float(0.0f, 1.0f) > dl->rate ? 1.0f : 0.0f;
        }
    }

    mat_had(y->m, x.m, dl->mask.m);
}

void dropout_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == MAT);

    assert(dy.type == MAT);
    assert(dy.m.rows == dl->x_rows);
    assert(dy.m.cols == dl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(dl->x_rows, dl->batch_size);

    mat_had(dx->m, dy.m, dl->mask.m);
}

void dropout_2D_destroy(layer l)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == MAT);

    free(dl->mask.m.vals);

    free(dl);
}

void dropout_3D_forward(layer l, tens x, tens *y)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS3D);

    assert(x.type == TENS3D);
    assert(x.t3.rows == dl->x_rows);
    assert(x.t3.cols == dl->x_cols);
    assert(x.t3.depth == dl->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(dl->x_rows, dl->x_cols, dl->batch_size);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < dl->x_rows; ++i) {
        for (int j = 0; j < dl->x_cols; ++j) {
            for (int k = 0; k < dl->batch_size; ++k) {
                tens3D_at(dl->mask.t3, i, j, k) =
                    rand_float(0.0f, 1.0f) > dl->rate ? 1.0f : 0.0f;
            }
        }
    }

    tens3D_had(y->t3, x.t3, dl->mask.t3);
}

void dropout_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS3D);

    assert(dy.type == TENS3D);
    assert(dy.t3.rows == dl->x_rows);
    assert(dy.t3.cols == dl->x_cols);
    assert(dy.t3.depth == dl->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(dl->x_rows, dl->x_cols, dl->batch_size);

    tens3D_had(dx->t3, dy.t3, dl->mask.t3);
}

void dropout_3D_destroy(layer l)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS3D);

    tens3D_destroy(dl->mask.t3);

    free(dl);
}

void dropout_4D_forward(layer l, tens x, tens *y)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS4D);

    assert(x.type == TENS4D);
    assert(x.t4.rows == dl->x_rows);
    assert(x.t4.cols == dl->x_cols);
    assert(x.t4.depth == dl->x_depth);
    assert(x.t4.batches == dl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(dl->x_rows, dl->x_cols,
                         dl->x_depth, dl->batch_size);

    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < dl->x_rows; ++i) {
        for (int j = 0; j < dl->x_cols; ++j) {
            for (int k = 0; k < dl->x_depth; ++k) {
                for (int l = 0; l < dl->batch_size; ++l) {
                    tens4D_at(dl->mask.t4, i, j, k, l) =
                        rand_float(0.0f, 1.0f) > dl->rate ? 1.0f : 0.0f;
                }
            }
        }
    }

    tens4D_had(y->t4, x.t4, dl->mask.t4);
}

void dropout_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS4D);

    assert(dy.type == TENS4D);
    assert(dy.t4.rows == dl->x_rows);
    assert(dy.t4.cols == dl->x_cols);
    assert(dy.t4.depth == dl->x_depth);
    assert(dy.t4.batches == dl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(dl->x_rows, dl->x_cols,
                          dl->x_depth, dl->batch_size);

    tens4D_had(dx->t4, dy.t4, dl->mask.t4);
}

void dropout_4D_destroy(layer l)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dl->x_type == TENS4D);

    tens4D_destroy(dl->mask.t4);

    free(dl);
}
