#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "nn.h"

layer dense_layer_alloc(int x_size, int y_size,
                        int batch_size, actfunc activation)
{
    dense_layer *dl = malloc(sizeof(dense_layer));
    dl->x_size = x_size;
    dl->y_size = y_size;
    dl->batch_size = batch_size;
    dl->w = mat_alloc(y_size, x_size);
    dl->b = mat_alloc(y_size, 1);
    dl->x_cache = mat_alloc(x_size, batch_size);
    dl->z_cache = mat_alloc(y_size, batch_size);
    dl->activation = activation;

    layer l;
    l.type = DENSE;
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

void dense_forward(layer l, void *x, void **y)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_x = (mat *)x;

    assert(mat_x->rows == dl->x_size);
    assert(mat_x->cols == dl->batch_size);

    mat *mat_y = malloc(sizeof(mat));
    *mat_y = mat_alloc(dl->y_size, dl->batch_size);

    mat_copy(dl->x_cache, *mat_x);

    mat_dot(dl->z_cache, dl->w, dl->x_cache);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        for (int j = 0; j < dl->batch_size; ++j) {
            mat_at(dl->z_cache, i, j) += mat_at(dl->b, i, 0);
        }
    }

    switch (dl->activation) {
        case LIN:
            mat_func(*mat_y, dl->z_cache, lin);
            break;
        case SIG:
            mat_func(*mat_y, dl->z_cache, sig);
            break;
        case TANH:
            mat_func(*mat_y, dl->z_cache, tanhf);
            break;
        case RELU:
            mat_func(*mat_y, dl->z_cache, relu);
            break;
    }

    *y = mat_y;
}

void dense_backprop(layer l, void *dy, void **dx, float rate)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_dy = (mat *)dy;

    assert(mat_dy->rows == dl->y_size);
    assert(mat_dy->cols == dl->batch_size);

    mat *mat_dx = malloc(sizeof(mat));
    *mat_dx = mat_alloc(dl->x_size, dl->batch_size);

    mat dz = mat_alloc(dl->y_size, dl->batch_size);
    mat delta = mat_alloc(dl->y_size, dl->batch_size);

    switch (dl->activation) {
        case LIN:
            mat_func(dz, dl->z_cache, dlin);
            break;
        case SIG:
            mat_func(dz, dl->z_cache, dsig);
            break;
        case TANH:
            mat_func(dz, dl->z_cache, dtanh);
            break;
        case RELU:
            mat_func(dz, dl->z_cache, drelu);
            break;
    }

    mat_had(delta, *mat_dy, dz);

    mat x_T = mat_alloc(dl->batch_size, dl->x_size);
    mat_trans(x_T, dl->x_cache);

    mat dw = mat_alloc(dl->y_size, dl->x_size);
    mat_dot(dw, delta, x_T);

    mat db = mat_alloc(dl->y_size, 1);
    mat_fill(db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < dl->batch_size; ++j) {
            sum += mat_at(delta, i, j);
        }

        mat_at(db, i, 0) = sum;
    }

    mat_scale(dw, dw, 1.0 / dl->batch_size);
    mat_func(dw, dw, clip);
    mat_scale(dw, dw, rate);
    mat_sub(dl->w, dl->w, dw);

    mat_scale(db, db, 1.0 / dl->batch_size);
    mat_func(db, db, clip);
    mat_scale(db, db, rate);
    mat_sub(dl->b, dl->b, db);

    mat w_T = mat_alloc(dl->w.cols, dl->w.rows);
    mat_trans(w_T, dl->w);

    mat_dot(*mat_dx, w_T, delta);

    *dx = mat_dx;

    free(dz.vals);
    free(delta.vals);
    free(x_T.vals);
    free(dw.vals);
    free(db.vals);
    free(w_T.vals);
}

void dense_destroy(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    free(dl->w.vals);
    free(dl->b.vals);
    free(dl->x_cache.vals);
    free(dl->z_cache.vals);

    free(dl);
}

void dense_init(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    if (dl->activation == SIG || dl->activation == TANH) {
        mat_normal(dl->w, 0, sqrt(2.0 / (dl->x_size + dl->y_size)));
        mat_fill(dl->b, 0);
    }
    else {
        mat_normal(dl->w, 0, sqrt(2.0 / dl->x_size));
        mat_fill(dl->b, 0);
    }
}

void dense_print(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_print(dl->w);
    mat_print(dl->b);
}

void dense_save(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_save(dl->w, f);
    mat_save(dl->b, f);
}

void dense_load(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_load(dl->w, f);
    mat_load(dl->b, f);
}
