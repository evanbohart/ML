#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "nn.h"

layer dense_layer_alloc(int x_size, int y_size, int batch_size)
{
    dense_layer *dl = malloc(sizeof(dense_layer));
    dl->x_size = x_size;
    dl->y_size = y_size;
    dl->batch_size = batch_size;
    dl->w = mat_alloc(y_size, x_size);
    dl->b = mat_alloc(y_size, 1);
    dl->x_cache = mat_alloc(x_size, batch_size);

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

    assert(x.type == MAT);
    assert(x.m.rows == dl->x_size);
    assert(x.m.cols == dl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(dl->y_size, dl->batch_size);

    mat_copy(dl->x_cache, x.m);

    mat_dot(y->m, dl->w, x.m);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        for (int j = 0; j < dl->batch_size; ++j) {
            mat_at(y->m, i, j) += mat_at(dl->b, i, 0);
        }
    }
}

void dense_backprop(layer l, tens dy, tens *dx, float rate)
{
    dense_layer *dl = (dense_layer *)l.data;

    assert(dy.type == MAT);
    assert(dy.m.rows == dl->y_size);
    assert(dy.m.cols == dl->batch_size);

    dx->type = MAT;
    dx->m = mat_alloc(dl->x_size, dl->batch_size);

    mat x_T = mat_alloc(dl->batch_size, dl->x_size);
    mat_trans(x_T, dl->x_cache);

    mat w_T = mat_alloc(dl->w.cols, dl->w.rows);
    mat_trans(w_T, dl->w);

    mat dw = mat_alloc(dl->y_size, dl->x_size);
    mat_dot(dw, dy.m, x_T);

    mat db = mat_alloc(dl->y_size, 1);
    mat_fill(db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dl->y_size; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < dl->batch_size; ++j) {
            sum += mat_at(dy.m, i, j);
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

    mat_dot(dx->m, w_T, dy.m);

    free(x_T.vals);
    free(w_T.vals);
    free(dw.vals);
    free(db.vals);
}

void dense_destroy(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    free(dl->w.vals);
    free(dl->b.vals);
    free(dl->x_cache.vals);

    free(dl);
}

void dense_init(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

            mat_normal(dl->w, 0, sqrt(2.0 / dl->x_size));

    mat_fill(dl->b, 0);
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
