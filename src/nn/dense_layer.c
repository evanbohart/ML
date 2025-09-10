#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer dense_layer_alloc(int x_size, int y_size, int batch_size)
{
    dense_layer *dl = malloc(sizeof(dense_layer));

    dl->x_size = x_size;
    dl->x_batches = batch_size;

    dl->y_size = y_size;

    dl->w = tens2D_alloc(y_size, x_size);
    dl->b = tens2D_alloc(y_size, 1);

    dl->x_T = tens2D_alloc(batch_size, x_size);
    dl->w_T = tens2D_alloc(x_size, y_size);

    dl->dw = tens2D_alloc(y_size, x_size);
    dl->db = tens2D_alloc(y_size, 1);

    layer l;

    l.data = dl;

    l.forward = dense_forward;
    l.backprop = dense_backprop;
    l.destroy = dense_destroy;

    l.init = dense_init;
    l.print = dense_print;
    l.save = dense_save;
    l.load = dense_load;

    return l;
}

void dense_forward(layer l, tens x, tens *y)
{
    dense_layer *dl = (dense_layer *)l.data;

    assert(x.rows == dl->x_size);
    assert(x.cols == dl->x_batches);
    assert(x.depth == 1);
    assert(x.batches == 1);

    *y = tens2D_alloc(dl->y_size, dl->x_batches);

    tens_trans(dl->x_T, x);

    tens_dot(*y, dl->w, x);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        for (int j = 0; j < dl->x_batches; ++j) {
            tens2D_at(*y, i, j) += tens2D_at(dl->b, i, 0);
        }
    }
}

void dense_backprop(layer l, tens dy, tens *dx, float rate)
{
    dense_layer *dl = (dense_layer *)l.data;

    assert(dy.rows == dl->y_size);
    assert(dy.cols == dl->x_batches);

    *dx = tens2D_alloc(dl->x_size, dl->x_batches);

    tens_trans(dl->w_T, dl->w);
    tens_dot(*dx, dl->w_T, dy);

    tens_dot(dl->dw, dy, dl->x_T);

    tens_fill(dl->db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < dl->x_batches; ++j) {
            sum += tens2D_at(dy, i, j);
        }

        tens2D_at(dl->db, i, 0) = sum;
    }

    tens_scale(dl->dw, dl->dw, rate / dl->x_batches);
    tens_func(dl->dw, dl->dw, clip);
    tens_sub(dl->w, dl->w, dl->dw);

    tens_scale(dl->db, dl->db, rate / dl->x_batches);
    tens_func(dl->db, dl->db, clip);
    tens_sub(dl->b, dl->b, dl->db);
}

void dense_destroy(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_destroy(dl->w);
    tens_destroy(dl->b);

    tens_destroy(dl->x_T);
    tens_destroy(dl->w_T);

    tens_destroy(dl->dw);
    tens_destroy(dl->db);

    free(dl);
}

void dense_init(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_normal(dl->w, 0, sqrt(2.0 / dl->x_size));

    tens_fill(dl->b, 0);
}

void dense_print(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_print(dl->w);
    tens_print(dl->b);
}

void dense_save(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_save(dl->w, f);
    tens_save(dl->b, f);
}

void dense_load(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_load(dl->w, f);
    tens_load(dl->b, f);
}
