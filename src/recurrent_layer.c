#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer recurrent_layer_alloc(int x_size, int h_size, int y_size,
                            int batch_size, int steps,
                            actfunc activation_h, actfunc activation_y)
{
    recurrent_layer *rl = malloc(sizeof(recurrent_layer));

    rl->x_size = x_size;
    rl->h_size = h_size;
    rl->y_size = y_size;
    rl->batch_size = batch_size;
    rl->steps = steps;

    rl->w_x = mat_alloc(h_size, x_size);
    rl->w_h = mat_alloc(h_size, h_size);
    rl->w_y = mat_alloc(y_size, h_size);

    rl->b_h = mat_alloc(h_size, 1);
    rl->b_y = mat_alloc(y_size, 1);

    rl->h_0 = mat_alloc(h_size, 1);

    rl->activation_h = activation_h;
    rl->activation_y = activation_y;

    rl->x_cache = tens3D_alloc(x_size, batch_size, steps);
    rl->h_z_cache = tens3D_alloc(h_size, batch_size, steps);
    rl->h_cache = tens3D_alloc(h_size, batch_size, steps);
    rl->y_z_cache = tens3D_alloc(y_size, batch_size, steps);

    layer l;

    l.type = RECURRENT;
    l.data = rl;
    l.forward = recurrent_forward;
    l.backprop = recurrent_backprop;
    l.destroy = recurrent_destroy;
    l.init = recurrent_init;
    l.print = recurrent_print;
    l.save = recurrent_save;
    l.load = recurrent_load;

    return l;
}

void recurrent_forward(layer l, void *x, void **y)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;
    tens3D *tens3D_x = (tens3D *)x;

    assert(tens3D_x->rows == rl->x_size);
    assert(tens3D_x->cols == rl->batch_size);
    assert(tens3D_x->depth == rl->steps);

    tens3D *tens3D_y = malloc(sizeof(tens3D));
    *tens3D_y = tens3D_alloc(rl->y_size, rl->batch_size, rl->steps);

    mat w_x_dot_x = mat_alloc(rl->h_size, rl->batch_size);
    mat w_h_dot_h_prev = mat_alloc(rl->h_size, rl->batch_size);

    tens3D_copy(rl->x_cache, *tens3D_x);

    for (int i = 0; i < rl->steps; ++i) {
        mat_dot(w_x_dot_x, rl->w_x, tens3D_x->mats[i]);

        if (i > 0) {
            mat_dot(w_h_dot_h_prev, rl->w_h, rl->h_cache.mats[i - 1]);
        }
        else {
            mat broadcasted = mat_alloc(rl->h_size, rl->batch_size);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < rl->h_size; ++j) {
                for (int k = 0; k < rl->batch_size; ++k) {
                    mat_at(broadcasted, j, k) = mat_at(rl->h_0, j, 0);
                }
            }

            mat_dot(w_h_dot_h_prev, rl->w_h, broadcasted);
 
            free(broadcasted.vals);
        }

        mat_add(rl->h_z_cache.mats[i], w_x_dot_x, w_h_dot_h_prev);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < rl->h_size; ++j) {
            for (int k = 0; k < rl->batch_size; ++k) {
                tens3D_at(rl->h_z_cache, j, k, i) += mat_at(rl->b_h, j, 0);
            }
        }

        switch (rl->activation_h) {
            case LIN:
                mat_func(rl->h_cache.mats[i], rl->h_z_cache.mats[i], lin);
                break;
            case SIG:
                mat_func(rl->h_cache.mats[i], rl->h_z_cache.mats[i], sig);
                break;
            case TANH:
                mat_func(rl->h_cache.mats[i], rl->h_z_cache.mats[i], tanhf);
                break;
            case RELU:
                mat_func(rl->h_cache.mats[i], rl->h_z_cache.mats[i], relu);
                break;
        }

        mat_dot(rl->y_z_cache.mats[i], rl->w_y, rl->h_cache.mats[i]);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < rl->y_size; ++j) {
            for (int k = 0; k < rl->batch_size; ++k) {
                tens3D_at(rl->y_z_cache, j, k, i) += mat_at(rl->b_y, j, 0);
            }
        }

        switch (rl->activation_y) {
            case LIN:
                mat_func(tens3D_y->mats[i], rl->y_z_cache.mats[i], lin);
                break;
            case SIG:
                mat_func(tens3D_y->mats[i], rl->y_z_cache.mats[i], sig);
                break;
            case TANH:
                mat_func(tens3D_y->mats[i], rl->y_z_cache.mats[i], tanhf);
                break;
            case RELU:
                mat_func(tens3D_y->mats[i], rl->y_z_cache.mats[i], relu);
                break;
        }
    }

    *y = tens3D_y;

    free(w_x_dot_x.vals);
    free(w_h_dot_h_prev.vals);
}

void recurrent_backprop(layer l, void *dy, void **dx, float rate)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;
    tens3D *tens3D_dy = (tens3D *)dy;

    assert(tens3D_dy->rows == rl->y_size);
    assert(tens3D_dy->cols == rl->batch_size);
    assert(tens3D_dy->depth == rl->steps);

    tens3D *tens3D_dx = malloc(sizeof(tens3D));
    *tens3D_dx = tens3D_alloc(rl->x_size, rl->batch_size, rl->steps);

    tens3D delta_h = tens3D_alloc(rl->h_size, rl->batch_size, rl->steps);
    tens3D delta_y = tens3D_alloc(rl->y_size, rl->batch_size, rl->steps);

    tens3D x_T = tens3D_alloc(rl->batch_size, rl->x_size, rl->steps);
    tens3D_trans(x_T, rl->x_cache);
    tens3D h_T = tens3D_alloc(rl->batch_size, rl->h_size, rl->steps);
    tens3D_trans(h_T, rl->h_cache);
    mat h_0_T = mat_alloc(1, rl->h_size);
    mat_trans(h_0_T, rl->h_0);

    tens3D dz_h = tens3D_alloc(rl->h_size, rl->batch_size, rl->steps);

    switch (rl->activation_h) {
        case LIN:
            tens3D_func(dz_h, rl->h_z_cache, dlin);
            break;
        case SIG:
            tens3D_func(dz_h, rl->h_z_cache, dsig);
            break;
        case TANH:
            tens3D_func(dz_h, rl->h_z_cache, dtanh);
            break;
        case RELU:
            tens3D_func(dz_h, rl->h_z_cache, drelu);
            break;
    }

    tens3D dz_y = tens3D_alloc(rl->y_size, rl->batch_size, rl->steps);

    switch (rl->activation_y) {
        case LIN:
            tens3D_func(dz_y, rl->y_z_cache, dlin);
            break;
        case SIG:
            tens3D_func(dz_y, rl->y_z_cache, dsig);
            break;
        case TANH:
            tens3D_func(dz_y, rl->y_z_cache, dtanh);
            break;
        case RELU:
            tens3D_func(dz_y, rl->y_z_cache, drelu);
            break;
    }


    tens3D_had(delta_y, *tens3D_dy, dz_y);

    mat w_x_T = mat_alloc(rl->x_size, rl->h_size);
    mat_trans(w_x_T, rl->w_x);
    mat w_h_T = mat_alloc(rl->h_size, rl->h_size);
    mat_trans(w_h_T, rl->w_h);
    mat w_y_T = mat_alloc(rl->h_size, rl->y_size);
    mat_trans(w_y_T, rl->w_y);

    mat temp = mat_alloc(rl->h_size, rl->batch_size);

    mat dw_x_current = mat_alloc(rl->h_size, rl->x_size);
    mat dw_h_current = mat_alloc(rl->h_size, rl->h_size);
    mat dw_y_current = mat_alloc(rl->y_size, rl->h_size);

    mat dw_x = mat_alloc(rl->h_size, rl->x_size);
    mat_fill(dw_x, 0);
    mat dw_h = mat_alloc(rl->h_size, rl->h_size);
    mat_fill(dw_h, 0);
    mat dw_y = mat_alloc(rl->y_size, rl->h_size);
    mat_fill(dw_y, 0);

    mat db_h_current = mat_alloc(rl->h_size, 1);
    mat db_y_current = mat_alloc(rl->y_size, 1);

    mat db_h = mat_alloc(rl->h_size, 1);
    mat_fill(db_h, 0);
    mat db_y = mat_alloc(rl->y_size, 1);
    mat_fill(db_y, 0);

    mat dh_0 = mat_alloc(rl->h_size, 1);
    mat_fill(dh_0, 0);

    for (int i = rl->steps - 1; i >= 0; --i) {
        mat_dot(dw_y_current, delta_y.mats[i], h_T.mats[i]);

        if (i < rl->steps - 1) {
            mat_dot(delta_h.mats[i], w_y_T, delta_y.mats[i]);
            mat_dot(temp, w_h_T, delta_h.mats[i + 1]);
            mat_add(delta_h.mats[i], delta_h.mats[i], temp);
            mat_had(delta_h.mats[i], delta_h.mats[i], dz_h.mats[i]);
        }
        else {
            mat_dot(delta_h.mats[i], w_y_T, delta_y.mats[i]);
            mat_had(delta_h.mats[i], delta_h.mats[i], dz_h.mats[i]);
        }

        mat_dot(tens3D_dx->mats[i], w_x_T, delta_h.mats[i]);

        mat_dot(dw_x_current, delta_h.mats[i], x_T.mats[i]);

        if (i > 0) {
            mat_dot(dw_h_current, delta_h.mats[i], h_T.mats[i]);
        }
        else {
            mat broadcasted = mat_alloc(rl->batch_size, rl->h_size);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < rl->batch_size; ++j) {
                for (int k = 0; k < rl->h_size; ++k) {
                    mat_at(broadcasted, j, k) = mat_at(h_0_T, 0, k);
                }
            }

            mat_dot(dw_h_current, delta_h.mats[i], broadcasted);

            free(broadcasted.vals);

            mat_dot(temp, w_h_T, delta_h.mats[0]);

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < rl->h_size; ++j) {
                float sum = 0.0f;

                for (int k = 0; k < rl->batch_size; ++k) {
                    sum += mat_at(temp, j, k);
                }

                mat_at(dh_0, j, 0) = sum;
            }
        }

        mat_add(dw_x, dw_x, dw_x_current);
        mat_add(dw_h, dw_h, dw_h_current);
        mat_add(dw_y, dw_y, dw_y_current);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < rl->h_size; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < rl->batch_size; ++k) {
                sum += mat_at(delta_h.mats[i], j, k);
            }

            mat_at(db_h_current, j, 0) = sum;
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < rl->y_size; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < rl->batch_size; ++k) {
                sum += mat_at(delta_y.mats[i], j, k);
            }

            mat_at(db_y_current, j, 0) = sum;
        }

        mat_add(db_h, db_h, db_h_current);
        mat_add(db_y, db_y, db_y_current);
    }

    *dx = tens3D_dx;

    mat_scale(dw_x, dw_x, 1.0f / rl->batch_size);
    mat_func(dw_x, dw_x, clip);
    mat_scale(dw_x, dw_x, rate);
    mat_sub(rl->w_x, rl->w_x, dw_x);

    mat_scale(dw_h, dw_h, 1.0f / rl->batch_size);
    mat_func(dw_h, dw_h, clip);
    mat_scale(dw_h, dw_h, rate);
    mat_sub(rl->w_h, rl->w_h, dw_h);

    mat_scale(dw_y, dw_y, 1.0f / rl->batch_size);
    mat_func(dw_y, dw_y, clip);
    mat_scale(dw_y, dw_y, rate);
    mat_sub(rl->w_y, rl->w_y, dw_y);

    mat_scale(db_h, db_h, 1.0f / rl->batch_size);
    mat_func(db_h, db_h, clip);
    mat_scale(db_h, db_h, rate);
    mat_sub(rl->b_h, rl->b_h, db_h);

    mat_scale(db_y, db_y, 1.0f / rl->batch_size);
    mat_func(db_y, db_y, clip);
    mat_scale(db_y, db_y, rate);
    mat_sub(rl->b_y, rl->b_y, db_y);

    mat_scale(dh_0, dh_0, 1.0f / rl->batch_size);
    mat_func(dh_0, dh_0, clip);
    mat_scale(dh_0, dh_0, rate);
    mat_sub(rl->h_0, rl->h_0, dh_0);

    tens3D_destroy(delta_h);
    tens3D_destroy(delta_y);
    tens3D_destroy(x_T);
    tens3D_destroy(h_T);
    free(h_0_T.vals);
    tens3D_destroy(dz_h);
    tens3D_destroy(dz_y);
    free(w_x_T.vals);
    free(w_h_T.vals);
    free(w_y_T.vals);
    free(temp.vals);
    free(dw_x_current.vals);
    free(dw_h_current.vals);
    free(dw_y_current.vals);
    free(dw_x.vals);
    free(dw_h.vals);
    free(dw_y.vals);
    free(db_h_current.vals);
    free(db_y_current.vals);
    free(db_h.vals);
    free(db_y.vals);
    free(dh_0.vals);
}

void recurrent_destroy(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    free(rl->w_x.vals);
    free(rl->w_h.vals);
    free(rl->w_y.vals);
    free(rl->b_h.vals);
    free(rl->b_y.vals);
    free(rl->h_0.vals);

    tens3D_destroy(rl->x_cache);
    tens3D_destroy(rl->h_z_cache);
    tens3D_destroy(rl->h_cache);
    tens3D_destroy(rl->y_z_cache);

    free(rl);
}

void recurrent_init(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    if (rl->activation_h == SIG || rl->activation_y == TANH) {
        mat_normal(rl->w_x, 0, sqrt(2.0f / (rl->x_size + rl->h_size)));
        mat_normal(rl->w_h, 0, sqrt(2.0f / (rl->h_size + rl->h_size)));
    }
    else {
        mat_normal(rl->w_x, 0, sqrt(2.0f / rl->x_size));
        mat_normal(rl->w_h, 0, sqrt(2.0f / rl->h_size));
    }

    if (rl->activation_y == SIG || rl->activation_y == TANH) {
        mat_normal(rl->w_y, 0, sqrt(2.0f / (rl->h_size + rl->y_size)));
    }
    else {
        mat_normal(rl->w_y, 0, sqrt(2.0f / rl->h_size));
    }

    mat_fill(rl->b_h, 0);
    mat_fill(rl->b_y, 0);
    mat_fill(rl->h_0, 0);
}

void recurrent_print(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_print(rl->w_x);
    mat_print(rl->w_h);
    mat_print(rl->w_y);
    mat_print(rl->b_h);
    mat_print(rl->b_y);
    mat_print(rl->h_0);
}

void recurrent_save(layer l, FILE *f)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_save(rl->w_x, f);
    mat_save(rl->w_h, f);
    mat_save(rl->w_y, f);
    mat_save(rl->b_h, f);
    mat_save(rl->b_y, f);
    mat_save(rl->h_0, f);
}

void recurrent_load(layer l, FILE *f)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_load(rl->w_x, f);
    mat_load(rl->w_h, f);
    mat_load(rl->w_y, f);
    mat_load(rl->b_h, f);
    mat_load(rl->b_y, f);
    mat_load(rl->h_0, f);
}
