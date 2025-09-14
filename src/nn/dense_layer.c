#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer dense_layer_alloc(int x_r, int y_r, int x_b)
{
    dense_layer *dl = malloc(sizeof(dense_layer));

    dl->x_r = x_r;
    dl->x_b = x_b;

    dl->y_r = y_r;

    dl->w = tens_alloc(y_r, x_r, 1, 1);
    dl->b = tens_alloc(y_r, 1, 1, 1);

    dl->x_reshaped = tens_alloc(x_r, x_b, 1, 1);
    dl->dot = tens_alloc(y_r, x_b, 1, 1);

    dl->x_reshaped_T = tens_alloc(x_b, x_r, 1, 1);
    dl->w_T = tens_alloc(x_r, y_r, 1, 1);

    dl->dy_reshaped = tens_alloc(y_r, x_b, 1, 1);
    dl->dx_reshaped = tens_alloc(x_r, x_b, 1, 1);

    dl->dw = tens_alloc(y_r, x_r, 1, 1);
    dl->db = tens_alloc(y_r, 1, 1, 1);

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

    assert(x.dims[R] == dl->x_r);
    assert(x.dims[C] == 1);
    assert(x.dims[D] == 1);
    assert(x.dims[B] == dl->x_b);

    *y = tens_alloc(dl->y_r, 1, 1, dl->x_b);

    tens_reshape(dl->x_reshaped, x);

    int perm[4] = { C, R, D, B };
    tens_trans(dl->x_reshaped_T, dl->x_reshaped, perm);

    tens_dot(dl->dot, dl->w, dl->x_reshaped);

    tens_reshape(y, dl->dot);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->y_r; ++i) {
        for (int j = 0; j < dl->x_b; ++j) {
            tens_at(*y, i, 0, 0, j) = tens_at(dl->b, i, 0, 0, 0);
        }
    }
}

void dense_backprop(layer l, tens dy, tens *dx, float rate)
{
    dense_layer *dl = (dense_layer *)l.data;

    assert(dy.dims[R] == dl->y_r);
    assert(dy.dims[C] == 1);
    assert(dy.dims[D] == 1);
    assert(dy.dims[B] == dl->x_b);

    *dx = tens_alloc(dl->x_r, 1, 1, dl->x_b);

    int perm[4] = { C, R, D, B };
    tens_trans(dl->w_T, dl->w, perm);

    tens_reshape(dl->dy_reshaped, dy);

    tens_dot(dl->dx_reshaped, dl->w_T, dl->dy_reshaped);
    tens_reshape(*dx, dl->dx_reshaped);

    tens_dot(dl->dw, dl->dy_reshaped, dl->x_reshaped_T);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dl->y_r; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < dl->x_b; ++j) {
            sum += tens_at(dy, i, 0, 0, j);
        }

        tens_at(dl->db, i, 0, 0, 0) = sum;
    }

    tens_scale(dl->dw, dl->dw, rate / dl->x_b);
    tens_func(dl->dw, dl->dw, clip);
    tens_sub(dl->w, dl->w, dl->dw);

    tens_scale(dl->db, dl->db, rate / dl->x_b);
    tens_func(dl->db, dl->db, clip);
    tens_sub(dl->b, dl->b, dl->db);
}

void dense_destroy(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_destroy(dl->w);
    tens_destroy(dl->b);

    tens_destroy(dl->x_reshaped);
    tens_destroy(dl->dot);

    tens_destroy(dl->x_reshaped_T);
    tens_destroy(dl->w_T);

    tens_destroy(dl->dy_reshaped);
    tens_destroy(dl->dx_reshaped);

    tens_destroy(dl->dw);
    tens_destroy(dl->db);

    free(dl);
}

void dense_init(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    tens_normal(dl->w, 0, sqrt(2.0 / dl->x_r));
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
