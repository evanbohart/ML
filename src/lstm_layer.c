#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer lstm_layer_alloc(int x_size, int h_size, int y_size,
                       int batch_size, int steps, actfunc activation)
{
    lstm_layer *ll = malloc(sizeof(lstm_layer));

    ll->x_size = x_size;
    ll->h_size = h_size;
    ll->y_size = y_size;
    ll->batch_size = batch_size;
    ll->steps = steps;

    ll->w_x_i = mat_alloc(h_size, x_size);
    ll->w_x_f = mat_alloc(h_size, x_size);
    ll->w_x_o = mat_alloc(h_size, x_size);
    ll->w_x_cc = mat_alloc(h_size, x_size);
    ll->w_h_i = mat_alloc(h_size, h_size);
    ll->w_h_f = mat_alloc(h_size, h_size);
    ll->w_h_o = mat_alloc(h_size, h_size);
    ll->w_h_cc = mat_alloc(h_size, h_size);
    ll->w_y = mat_alloc(y_size, h_size);

    ll->b_i = mat_alloc(h_size, 1);
    ll->b_f = mat_alloc(h_size, 1);
    ll->b_o = mat_alloc(h_size, 1);
    ll->b_cc = mat_alloc(h_size, 1);
    ll->b_y = mat_alloc(y_size, 1);

    ll->h_0 = mat_alloc(h_size, 1);
    ll->c_0 = mat_alloc(h_size, 1);

    ll->x_cache = tens3D_alloc(x_size, batch_size, steps);
    ll->h_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->c_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->i_a_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->f_a_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->o_a_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->cc_a_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->i_z_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->f_z_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->o_z_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->cc_z_cache = tens3D_alloc(h_size, batch_size, steps);
    ll->y_z_cache = tens3D_alloc(y_size, batch_size, steps);

    ll->activation = activation;

    layer l;

    l.type = LSTM;
    l.data = ll;
    l.forward = lstm_forward;
    l.backprop = lstm_backprop;
    l.destroy = lstm_destroy;
    l.init = lstm_init;
    l.print = lstm_print;
    l.save = lstm_save;
    l.load = lstm_load;

    return l;
}

void lstm_forward(layer l, void *x, void **y)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    tens3D *tens3D_x = (tens3D *)x;

    assert(tens3D_x->rows == ll->x_size);
    assert(tens3D_x->cols == ll->batch_size);
    assert(tens3D_x->depth == ll->steps);

    tens3D *tens3D_y = malloc(sizeof(tens3D));
    *tens3D_y = tens3D_alloc(ll->y_size, ll->batch_size, ll->steps);

    tens3D_copy(ll->x_cache, *tens3D_x);

    mat w_x_i_dot_x = mat_alloc(ll->h_size, ll->batch_size);
    mat w_x_f_dot_x = mat_alloc(ll->h_size, ll->batch_size);
    mat w_x_o_dot_x = mat_alloc(ll->h_size, ll->batch_size);
    mat w_x_cc_dot_x = mat_alloc(ll->h_size, ll->batch_size);

    mat w_h_i_dot_h_prev = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_f_dot_h_prev = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_o_dot_h_prev = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_cc_dot_h_prev = mat_alloc(ll->h_size, ll->batch_size);

    mat i_had_cc = mat_alloc(ll->h_size, ll->batch_size);
    mat f_had_cprev = mat_alloc(ll->h_size, ll->batch_size);

    mat b_i_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat b_f_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat b_o_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat b_cc_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat b_y_broadcasted = mat_alloc(ll->y_size, ll->batch_size);

    mat h_0_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat c_0_broadcasted = mat_alloc(ll->h_size, ll->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ll->h_size; ++i) {
        for (int j = 0; j < ll->batch_size; ++j) {
            mat_at(b_i_broadcasted, i, j) = mat_at(ll->b_i, i, 0);
            mat_at(b_f_broadcasted, i, j) = mat_at(ll->b_f, i, 0);
            mat_at(b_o_broadcasted, i, j) = mat_at(ll->b_o, i, 0);
            mat_at(b_cc_broadcasted, i, j) = mat_at(ll->b_cc, i, 0);

            mat_at(h_0_broadcasted, i, j) = mat_at(ll->h_0, i, 0);
            mat_at(c_0_broadcasted, i, j) = mat_at(ll->c_0, i, 0);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ll->y_size; ++i) {
        for (int j = 0; j < ll->batch_size; ++j) {
            mat_at(b_y_broadcasted, i, j) = mat_at(ll->b_y, i, 0);
        }
    }

    for (int i = 0; i < ll->steps; ++i) {
        mat_dot(w_x_i_dot_x, ll->w_x_i, ll->x_cache.mats[i]);
        mat_dot(w_x_f_dot_x, ll->w_x_f, ll->x_cache.mats[i]);
        mat_dot(w_x_o_dot_x, ll->w_x_o, ll->x_cache.mats[i]);
        mat_dot(w_x_cc_dot_x, ll->w_x_cc, ll->x_cache.mats[i]);

        if (i > 0) {
            mat_dot(w_h_i_dot_h_prev, ll->w_h_i, ll->h_cache.mats[i - 1]);
            mat_dot(w_h_f_dot_h_prev, ll->w_h_f, ll->h_cache.mats[i - 1]);
            mat_dot(w_h_o_dot_h_prev, ll->w_h_o, ll->h_cache.mats[i - 1]);
            mat_dot(w_h_cc_dot_h_prev, ll->w_h_cc, ll->h_cache.mats[i - 1]);
        }
        else {
            mat_dot(w_h_i_dot_h_prev, ll->w_h_i, h_0_broadcasted);
            mat_dot(w_h_f_dot_h_prev, ll->w_h_f, h_0_broadcasted);
            mat_dot(w_h_o_dot_h_prev, ll->w_h_o, h_0_broadcasted);
            mat_dot(w_h_cc_dot_h_prev, ll->w_h_cc, h_0_broadcasted);
        }

        mat_add(ll->i_z_cache.mats[i], w_x_i_dot_x, w_h_i_dot_h_prev);
        mat_add(ll->f_z_cache.mats[i], w_x_f_dot_x, w_h_f_dot_h_prev);
        mat_add(ll->o_z_cache.mats[i], w_x_o_dot_x, w_h_o_dot_h_prev);
        mat_add(ll->cc_z_cache.mats[i], w_x_cc_dot_x, w_h_cc_dot_h_prev);

        mat_add(ll->i_z_cache.mats[i], ll->i_z_cache.mats[i], b_i_broadcasted);
        mat_add(ll->f_z_cache.mats[i], ll->f_z_cache.mats[i], b_f_broadcasted);
        mat_add(ll->o_z_cache.mats[i], ll->o_z_cache.mats[i], b_o_broadcasted);
        mat_add(ll->cc_z_cache.mats[i], ll->cc_z_cache.mats[i], b_cc_broadcasted);

        mat_func(ll->i_a_cache.mats[i], ll->i_z_cache.mats[i], sig);
        mat_func(ll->f_a_cache.mats[i], ll->f_z_cache.mats[i], sig);
        mat_func(ll->o_a_cache.mats[i], ll->o_z_cache.mats[i], sig);
        mat_func(ll->cc_a_cache.mats[i], ll->cc_z_cache.mats[i], tanhf);

        mat_had(i_had_cc, ll->i_a_cache.mats[i], ll->cc_a_cache.mats[i]);

        if (i > 0) {
            mat_had(f_had_cprev, ll->f_a_cache.mats[i], ll->c_cache.mats[i - 1]);
        }
        else {
            mat_had(f_had_cprev, ll->f_a_cache.mats[i], c_0_broadcasted);
        }

        mat_add(ll->c_cache.mats[i], i_had_cc, f_had_cprev);
        mat_func(ll->h_cache.mats[i], ll->c_cache.mats[i], tanhf);
        mat_had(ll->h_cache.mats[i], ll->h_cache.mats[i], ll->o_a_cache.mats[i]);

        mat_dot(ll->y_z_cache.mats[i], ll->w_y, ll->h_cache.mats[i]);
        mat_add(ll->y_z_cache.mats[i], ll->y_z_cache.mats[i], b_y_broadcasted);

        switch (ll->activation) {
            case LIN:
                mat_func(tens3D_y->mats[i], ll->y_z_cache.mats[i], lin);
                break;
            case SIG:
                mat_func(tens3D_y->mats[i], ll->y_z_cache.mats[i], sig);
                break;
            case TANH:
                mat_func(tens3D_y->mats[i], ll->y_z_cache.mats[i], tanhf);
                break;
            case RELU:
                mat_func(tens3D_y->mats[i], ll->y_z_cache.mats[i], relu);
                break;
        }
    }

    *y = tens3D_y;

    free(w_x_i_dot_x.vals);
    free(w_x_f_dot_x.vals);
    free(w_x_o_dot_x.vals);
    free(w_x_cc_dot_x.vals);
    free(w_h_i_dot_h_prev.vals);
    free(w_h_f_dot_h_prev.vals);
    free(w_h_o_dot_h_prev.vals);
    free(w_h_cc_dot_h_prev.vals);

    free(i_had_cc.vals);
    free(f_had_cprev.vals);

    free(b_i_broadcasted.vals);
    free(b_f_broadcasted.vals);
    free(b_o_broadcasted.vals);
    free(b_cc_broadcasted.vals);
    free(b_y_broadcasted.vals);

    free(h_0_broadcasted.vals);
    free(c_0_broadcasted.vals);
}

void lstm_backprop(layer l, void *dy, void **dx, float rate)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    tens3D *tens3D_dy = (tens3D *)dy;

    assert(tens3D_dy->rows == ll->y_size);
    assert(tens3D_dy->cols == ll->batch_size);
    assert(tens3D_dy->depth == ll->steps);

    tens3D *tens3D_dx = malloc(sizeof(tens3D));
    *tens3D_dx = tens3D_alloc(ll->x_size, ll->batch_size, ll->steps);

    tens3D dh = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D dc = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);

    tens3D di_a = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D df_a = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D do_a = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D dcc_a = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);

    tens3D di_z = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D df_z = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D do_z = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D dcc_z = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D dy_z = tens3D_alloc(ll->y_size, ll->batch_size, ll->steps);

    tens3D i_z_dsig = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(i_z_dsig, ll->i_z_cache, dsig);
    tens3D f_z_dsig = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(f_z_dsig, ll->f_z_cache, dsig);
    tens3D o_z_dsig = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(o_z_dsig, ll->o_z_cache, dsig);
    tens3D cc_z_dtanh = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(cc_z_dtanh, ll->cc_z_cache, dtanh);

    tens3D y_z_da = tens3D_alloc(ll->y_size, ll->batch_size, ll->steps);

    switch (ll->activation) {
        case LIN:
            tens3D_func(y_z_da, ll->y_z_cache, dlin);
            break;
        case SIG:
            tens3D_func(y_z_da, ll->y_z_cache, dsig);
            break;
        case TANH:
            tens3D_func(y_z_da, ll->y_z_cache, dtanh);
            break;
        case RELU:
            tens3D_func(y_z_da, ll->y_z_cache, drelu);
            break;
    }

    tens3D c_tanh = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(c_tanh, ll->c_cache, tanhf);
    tens3D c_dtanh = tens3D_alloc(ll->h_size, ll->batch_size, ll->steps);
    tens3D_func(c_dtanh, ll->c_cache, dtanh);

    tens3D x_T = tens3D_alloc(ll->batch_size, ll->x_size, ll->steps);
    tens3D_trans(x_T, ll->x_cache);
    tens3D h_T = tens3D_alloc(ll->batch_size, ll->h_size, ll->steps);
    tens3D_trans(h_T, ll->h_cache);

    mat w_x_i_T = mat_alloc(ll->x_size, ll->h_size);
    mat_trans(w_x_i_T, ll->w_x_i);
    mat w_x_f_T = mat_alloc(ll->x_size, ll->h_size);
    mat_trans(w_x_f_T, ll->w_x_f);
    mat w_x_o_T = mat_alloc(ll->x_size, ll->h_size);
    mat_trans(w_x_o_T, ll->w_x_o);
    mat w_x_cc_T = mat_alloc(ll->x_size, ll->h_size);
    mat_trans(w_x_cc_T, ll->w_x_cc);
    mat w_h_i_T = mat_alloc(ll->h_size, ll->h_size);
    mat_trans(w_h_i_T, ll->w_h_i);
    mat w_h_f_T = mat_alloc(ll->h_size, ll->h_size);
    mat_trans(w_h_f_T, ll->w_h_f);
    mat w_h_o_T = mat_alloc(ll->h_size, ll->h_size);
    mat_trans(w_h_o_T, ll->w_h_o);
    mat w_h_cc_T = mat_alloc(ll->h_size, ll->h_size);
    mat_trans(w_h_cc_T, ll->w_h_cc);
    mat w_y_T = mat_alloc(ll->h_size, ll->y_size);
    mat_trans(w_y_T, ll->w_y);

    mat dc_next_had_f_next = mat_alloc(ll->h_size, ll->batch_size);

    mat w_x_i_T_dot_di_z = mat_alloc(ll->x_size, ll->batch_size);
    mat w_x_f_T_dot_df_z = mat_alloc(ll->x_size, ll->batch_size);
    mat w_x_o_T_dot_do_z = mat_alloc(ll->x_size, ll->batch_size);
    mat w_x_cc_T_dot_dcc_z = mat_alloc(ll->x_size, ll->batch_size);

    mat w_h_i_T_dot_di_z_next = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_f_T_dot_df_z_next = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_o_T_dot_do_z_next = mat_alloc(ll->h_size, ll->batch_size);
    mat w_h_cc_T_dot_dcc_z_next = mat_alloc(ll->h_size, ll->batch_size);

    mat w_y_T_dot_dy_z = mat_alloc(ll->h_size, ll->batch_size);

    mat dw_x_i_current = mat_alloc(ll->h_size, ll->x_size);
    mat dw_x_f_current = mat_alloc(ll->h_size, ll->x_size);
    mat dw_x_o_current = mat_alloc(ll->h_size, ll->x_size); 
    mat dw_x_cc_current = mat_alloc(ll->h_size, ll->x_size);
    mat dw_h_i_current = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_f_current = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_o_current = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_cc_current = mat_alloc(ll->h_size, ll->h_size);
    mat dw_y_current = mat_alloc(ll->y_size, ll->h_size);

    mat dw_x_i = mat_alloc(ll->h_size, ll->x_size);
    mat dw_x_f = mat_alloc(ll->h_size, ll->x_size);
    mat dw_x_o = mat_alloc(ll->h_size, ll->x_size);
    mat dw_x_cc = mat_alloc(ll->h_size, ll->x_size);
    mat dw_h_i = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_f = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_o = mat_alloc(ll->h_size, ll->h_size);
    mat dw_h_cc = mat_alloc(ll->h_size, ll->h_size);
    mat dw_y = mat_alloc(ll->y_size, ll->h_size);

    mat db_i_current = mat_alloc(ll->h_size, 1);
    mat db_f_current = mat_alloc(ll->h_size, 1);
    mat db_o_current = mat_alloc(ll->h_size, 1);
    mat db_cc_current = mat_alloc(ll->h_size, 1);
    mat db_y_current = mat_alloc(ll->y_size, 1);

    mat db_i = mat_alloc(ll->h_size, 1);
    mat db_f = mat_alloc(ll->h_size, 1);
    mat db_o = mat_alloc(ll->h_size, 1);
    mat db_cc = mat_alloc(ll->h_size, 1);
    mat db_y = mat_alloc(ll->y_size, 1);

    mat dh_0 = mat_alloc(ll->h_size, 1);
    mat dc_0 = mat_alloc(ll->h_size, 1);

    mat h_0_broadcasted = mat_alloc(ll->h_size, ll->batch_size);
    mat c_0_broadcasted = mat_alloc(ll->h_size, ll->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ll->h_size; ++i) {
        for (int j = 0; j < ll->batch_size; ++j) {
            mat_at(h_0_broadcasted, i, j) = mat_at(ll->h_0, i, 0);
            mat_at(c_0_broadcasted, i, j) = mat_at(ll->c_0, i, 0);
        }
    }

    mat h_0_broadcasted_T = mat_alloc(ll->batch_size, ll->h_size);
    mat_trans(h_0_broadcasted_T, h_0_broadcasted);

    tens3D_had(dy_z, *tens3D_dy, y_z_da);

    for (int i = ll->steps - 1; i >= 0; --i) {
        mat_dot(dw_y_current, dy_z.mats[i], h_T.mats[i]);

        if (i < ll->steps - 1) {
            mat_dot(w_h_i_T_dot_di_z_next, w_h_i_T, di_z.mats[i + 1]);
            mat_dot(w_h_f_T_dot_df_z_next, w_h_f_T, df_z.mats[i + 1]);
            mat_dot(w_h_o_T_dot_do_z_next, w_h_o_T, do_z.mats[i + 1]);
            mat_dot(w_h_cc_T_dot_dcc_z_next, w_h_cc_T, dcc_z.mats[i + 1]);
            mat_dot(w_y_T_dot_dy_z, w_y_T, dy_z.mats[i]);

            mat_add(dh.mats[i], w_h_i_T_dot_di_z_next, w_h_f_T_dot_df_z_next);
            mat_add(dh.mats[i], dh.mats[i], w_h_o_T_dot_do_z_next);
            mat_add(dh.mats[i], dh.mats[i], w_h_cc_T_dot_dcc_z_next);
            mat_add(dh.mats[i], dh.mats[i], w_y_T_dot_dy_z);
        }
        else {
            mat_dot(dh.mats[i], w_y_T, dy_z.mats[i]);
        }

        mat_had(dc.mats[i], dh.mats[i], ll->o_a_cache.mats[i]);
        mat_had(dc.mats[i], dc.mats[i], c_dtanh.mats[i]);

        if (i < ll->steps - 1) {
            mat_had(dc_next_had_f_next, dc.mats[i + 1], ll->f_a_cache.mats[i + 1]);
            mat_add(dc.mats[i], dc.mats[i], dc_next_had_f_next);
        }

        mat_had(di_a.mats[i], dc.mats[i], ll->cc_a_cache.mats[i]);

        if (i > 0) {
            mat_had(df_a.mats[i], dc.mats[i], ll->c_cache.mats[i - 1]);
        }
        else {
            mat_had(df_a.mats[i], dc.mats[i], c_0_broadcasted);
        }

        mat_had(do_a.mats[i], dh.mats[i], c_tanh.mats[i]);
        mat_had(dcc_a.mats[i], dc.mats[i], ll->i_a_cache.mats[i]);

        mat_had(di_z.mats[i], di_a.mats[i], i_z_dsig.mats[i]);
        mat_had(df_z.mats[i], df_a.mats[i], f_z_dsig.mats[i]);
        mat_had(do_z.mats[i], do_a.mats[i], o_z_dsig.mats[i]);
        mat_had(dcc_z.mats[i], dcc_a.mats[i], cc_z_dtanh.mats[i]);

        // calculate gradient of inputs
        mat_dot(w_x_i_T_dot_di_z, w_x_i_T, di_z.mats[i]);
        mat_dot(w_x_f_T_dot_df_z, w_x_f_T, df_z.mats[i]);
        mat_dot(w_x_o_T_dot_do_z, w_x_o_T, do_z.mats[i]);
        mat_dot(w_x_cc_T_dot_dcc_z, w_x_cc_T, dcc_z.mats[i]);

        mat_add(tens3D_dx->mats[i], w_x_i_T_dot_di_z, w_x_f_T_dot_df_z);
        mat_add(tens3D_dx->mats[i], tens3D_dx->mats[i], w_x_o_T_dot_do_z);
        mat_add(tens3D_dx->mats[i], tens3D_dx->mats[i], w_x_cc_T_dot_dcc_z);

        mat_dot(dw_x_i_current, di_z.mats[i], x_T.mats[i]);
        mat_dot(dw_x_f_current, df_z.mats[i], x_T.mats[i]);
        mat_dot(dw_x_o_current, do_z.mats[i], x_T.mats[i]);
        mat_dot(dw_x_cc_current, dcc_z.mats[i], x_T.mats[i]);

        if (i > 0) {
            mat_dot(dw_h_i_current, di_z.mats[i], h_T.mats[i - 1]);
            mat_dot(dw_h_f_current, df_z.mats[i], h_T.mats[i - 1]);
            mat_dot(dw_h_o_current, do_z.mats[i], h_T.mats[i - 1]);
            mat_dot(dw_h_cc_current, dcc_z.mats[i], h_T.mats[i - 1]);
        }
        else {
            mat_dot(dw_h_i_current, di_z.mats[i], h_0_broadcasted_T);
            mat_dot(dw_h_f_current, df_z.mats[i], h_0_broadcasted_T);
            mat_dot(dw_h_o_current, do_z.mats[i], h_0_broadcasted_T);
            mat_dot(dw_h_cc_current, dcc_z.mats[i], h_0_broadcasted_T);
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ll->h_size; ++j) {
            float sum_di_z = 0.0f;
            float sum_df_z = 0.0f;
            float sum_do_z = 0.0f;
            float sum_dcc_z = 0.0f;

            for (int k = 0; k < ll->batch_size; ++k) {
                sum_di_z += tens3D_at(di_z, j, k, i);
                sum_df_z += tens3D_at(df_z, j, k, i);
                sum_do_z += tens3D_at(do_z, j, k, i);
                sum_dcc_z += tens3D_at(dcc_z, j, k, i);
            }

            mat_at(db_i_current, j, 0) = sum_di_z;
            mat_at(db_f_current, j, 0) = sum_df_z;
            mat_at(db_o_current, j, 0) = sum_do_z;
            mat_at(db_cc_current, j, 0) = sum_dcc_z;
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ll->y_size; ++j) {
            float sum_dy_z = 0.0f;

            for (int k = 0; k < ll->batch_size; ++k) {
                sum_dy_z += tens3D_at(dy_z, j, k, i);
            }

            mat_at(db_y_current, j, 0) = sum_dy_z;
        }

        if (i == 0) {
            for (int j = 0; j < ll->h_size; ++j) {
                float sum_dh = 0.0f;
                float sum_dc = 0.0f;

                for (int k = 0; k < ll->batch_size; ++k) {
                    sum_dh += tens3D_at(dh, j, k, 0);
                    sum_dc += tens3D_at(dc, j, k, 0);
                }

                mat_at(dh_0, j, 0) = sum_dh;
                mat_at(dc_0, j, 0) = sum_dc;
            }
        }

        mat_add(dw_x_i, dw_x_i, dw_x_i_current);
        mat_add(dw_x_f, dw_x_f, dw_x_f_current);
        mat_add(dw_x_o, dw_x_o, dw_x_o_current);
        mat_add(dw_x_cc, dw_x_cc, dw_x_cc_current);
        mat_add(dw_h_i, dw_h_i, dw_h_i_current);
        mat_add(dw_h_f, dw_h_f, dw_h_f_current);
        mat_add(dw_h_o, dw_h_o, dw_h_o_current);
        mat_add(dw_h_cc, dw_h_cc, dw_h_cc_current);
        mat_add(dw_y, dw_y, dw_y_current);

        mat_add(db_i, db_i, db_i_current);
        mat_add(db_f, db_f, db_f_current);
        mat_add(db_o, db_o, db_o_current);
        mat_add(db_cc, db_cc, db_cc_current);
        mat_add(db_y, db_y, db_y_current);
    }

    mat_scale(dw_x_i, dw_x_i, 1.0f / ll->batch_size);
    mat_func(dw_x_i, dw_x_i, clip);
    mat_scale(dw_x_i, dw_x_i, rate);
    mat_sub(ll->w_x_i, ll->w_x_i, dw_x_i);

    mat_scale(dw_x_f, dw_x_f, 1.0f / ll->batch_size);
    mat_func(dw_x_f, dw_x_f, clip);
    mat_scale(dw_x_f, dw_x_f, rate);
    mat_sub(ll->w_x_f, ll->w_x_f, dw_x_f);

    mat_scale(dw_x_o, dw_x_o, 1.0f / ll->batch_size);
    mat_func(dw_x_o, dw_x_o, clip);
    mat_scale(dw_x_o, dw_x_o, rate);
    mat_sub(ll->w_x_o, ll->w_x_o, dw_x_o);

    mat_scale(dw_x_cc, dw_x_cc, 1.0f / ll->batch_size);
    mat_func(dw_x_cc, dw_x_cc, clip);
    mat_scale(dw_x_cc, dw_x_cc, rate);
    mat_sub(ll->w_x_cc, ll->w_x_cc, dw_x_cc);

    mat_scale(dw_h_i, dw_h_i, 1.0f / ll->batch_size);
    mat_func(dw_h_i, dw_h_i, clip);
    mat_scale(dw_h_i, dw_h_i, rate);
    mat_sub(ll->w_h_i, ll->w_h_i, dw_h_i);

    mat_scale(dw_h_f, dw_h_f, 1.0f / ll->batch_size);
    mat_func(dw_h_f, dw_h_f, clip);
    mat_scale(dw_h_f, dw_h_f, rate);
    mat_sub(ll->w_h_f, ll->w_h_f, dw_h_f);

    mat_scale(dw_h_o, dw_h_o, 1.0f / ll->batch_size);
    mat_func(dw_h_o, dw_h_o, clip);
    mat_scale(dw_h_o, dw_h_o, rate);
    mat_sub(ll->w_h_o, ll->w_h_o, dw_h_o);

    mat_scale(dw_h_cc, dw_h_cc, 1.0f / ll->batch_size);
    mat_func(dw_h_cc, dw_h_cc, clip);
    mat_scale(dw_h_cc, dw_h_cc, rate);
    mat_sub(ll->w_h_cc, ll->w_h_cc, dw_h_cc);

    mat_scale(dw_y, dw_y, 1.0f / ll->batch_size);
    mat_func(dw_y, dw_y, clip);
    mat_scale(dw_y, dw_y, rate);
    mat_sub(ll->w_y, ll->w_y, dw_y);

    mat_scale(db_i, db_i, 1.0f / ll->batch_size);
    mat_func(db_i, db_i, clip);
    mat_scale(db_i, db_i, rate);
    mat_sub(ll->b_i, ll->b_i, db_i);

    mat_scale(db_f, db_f, 1.0f / ll->batch_size);
    mat_func(db_f, db_f, clip);
    mat_scale(db_f, db_f, rate);
    mat_sub(ll->b_f, ll->b_f, db_f);

    mat_scale(db_o, db_o, 1.0f / ll->batch_size);
    mat_func(db_o, db_o, clip);
    mat_scale(db_o, db_o, rate);
    mat_sub(ll->b_o, ll->b_o, db_o);

    mat_scale(db_cc, db_cc, 1.0f / ll->batch_size);
    mat_func(db_cc, db_cc, clip);
    mat_scale(db_cc, db_cc, rate);
    mat_sub(ll->b_cc, ll->b_cc, db_cc);

    mat_scale(dh_0, dh_0, 1.0f / ll->batch_size);
    mat_func(dh_0, dh_0, clip);
    mat_scale(dh_0, dh_0, rate);
    mat_sub(ll->h_0, ll->h_0, dh_0);

    mat_scale(dc_0, dc_0, 1.0f / ll->batch_size);
    mat_func(dc_0, dc_0, clip);
    mat_scale(dc_0, dc_0, rate);
    mat_sub(ll->c_0, ll->c_0, dc_0);

    *dx = tens3D_dx;

    tens3D_destroy(dh);
    tens3D_destroy(dc);

    tens3D_destroy(di_a);
    tens3D_destroy(df_a);
    tens3D_destroy(do_a);
    tens3D_destroy(dcc_a);

    tens3D_destroy(di_z);
    tens3D_destroy(df_z);
    tens3D_destroy(do_z);
    tens3D_destroy(dcc_z);
    tens3D_destroy(dy_z);

    tens3D_destroy(i_z_dsig);
    tens3D_destroy(f_z_dsig);
    tens3D_destroy(o_z_dsig);
    tens3D_destroy(cc_z_dtanh);
    tens3D_destroy(y_z_da);

    tens3D_destroy(c_tanh);
    tens3D_destroy(c_dtanh);

    tens3D_destroy(x_T);
    tens3D_destroy(h_T);

    free(w_x_i_T.vals);
    free(w_x_f_T.vals);
    free(w_x_o_T.vals);
    free(w_x_cc_T.vals);
    free(w_h_i_T.vals);
    free(w_h_f_T.vals);
    free(w_h_o_T.vals);
    free(w_h_cc_T.vals);
    free(w_y_T.vals);

    free(dc_next_had_f_next.vals);

    free(w_h_i_T_dot_di_z_next.vals);
    free(w_h_f_T_dot_df_z_next.vals);
    free(w_h_o_T_dot_do_z_next.vals);
    free(w_h_cc_T_dot_dcc_z_next.vals);

    free(w_y_T_dot_dy_z.vals);

    free(dw_x_i_current.vals);
    free(dw_x_f_current.vals);
    free(dw_x_o_current.vals);
    free(dw_x_cc_current.vals);
    free(dw_h_i_current.vals);
    free(dw_h_f_current.vals);
    free(dw_h_o_current.vals);
    free(dw_h_cc_current.vals);
    free(dw_y_current.vals);

    free(dw_x_i.vals);
    free(dw_x_f.vals);
    free(dw_x_o.vals);
    free(dw_x_cc.vals);
    free(dw_h_i.vals);
    free(dw_h_f.vals);
    free(dw_h_o.vals);
    free(dw_h_cc.vals);
    free(dw_y.vals);

    free(db_i_current.vals);
    free(db_f_current.vals);
    free(db_o_current.vals);
    free(db_cc_current.vals);
    free(db_y_current.vals);

    free(db_i.vals);
    free(db_f.vals);
    free(db_o.vals);
    free(db_cc.vals);
    free(db_y.vals);

    free(dh_0.vals);
    free(dc_0.vals);

    free(h_0_broadcasted.vals);
    free(c_0_broadcasted.vals);

    free(h_0_broadcasted_T.vals);
}

void lstm_destroy(layer l)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    free(ll->w_x_i.vals);
    free(ll->w_x_f.vals);
    free(ll->w_x_o.vals);
    free(ll->w_x_cc.vals);
    free(ll->w_h_i.vals);
    free(ll->w_h_f.vals);
    free(ll->w_h_o.vals);
    free(ll->w_h_cc.vals);
    free(ll->w_y.vals);

    free(ll->b_i.vals);
    free(ll->b_f.vals);
    free(ll->b_o.vals);
    free(ll->b_cc.vals);
    free(ll->b_y.vals);

    free(ll->h_0.vals);
    free(ll->c_0.vals);

    tens3D_destroy(ll->x_cache);
    tens3D_destroy(ll->h_cache);
    tens3D_destroy(ll->c_cache);
    tens3D_destroy(ll->i_a_cache);
    tens3D_destroy(ll->f_a_cache);
    tens3D_destroy(ll->o_a_cache);
    tens3D_destroy(ll->cc_a_cache);
    tens3D_destroy(ll->i_z_cache);
    tens3D_destroy(ll->f_z_cache);
    tens3D_destroy(ll->o_z_cache);
    tens3D_destroy(ll->cc_z_cache);

    free(ll);
}

void lstm_init(layer l)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    mat_normal(ll->w_x_i, 0.0f, sqrt(2.0f / (ll->x_size + ll->h_size)));
    mat_normal(ll->w_x_f, 0.0f, sqrt(2.0f / (ll->x_size + ll->h_size)));
    mat_normal(ll->w_x_o, 0.0f, sqrt(2.0f / (ll->x_size + ll->h_size)));
    mat_normal(ll->w_x_cc, 0.0f, sqrt(2.0f / (ll->x_size + ll->h_size)));
    mat_normal(ll->w_h_i, 0.0f, sqrt(2.0f / (ll->h_size + ll->h_size)));
    mat_normal(ll->w_h_f, 0.0f, sqrt(2.0f / (ll->h_size + ll->h_size)));
    mat_normal(ll->w_h_o, 0.0f, sqrt(2.0f / (ll->h_size + ll->h_size)));
    mat_normal(ll->w_h_cc, 0.0f, sqrt(2.0f / (ll->h_size + ll->h_size)));

    if (ll->activation == SIG || ll->activation == TANH) {
        mat_normal(ll->w_y, 0.0f, sqrt(2.0f / (ll->h_size + ll->y_size)));
    }
    else {
        mat_normal(ll->w_y, 0.0f, sqrt(2.0f / ll->h_size));
    }

    mat_fill(ll->b_i, 0.0f);
    mat_fill(ll->b_f, 0.0f);
    mat_fill(ll->b_o, 0.0f);
    mat_fill(ll->b_cc, 0.0f);
    mat_fill(ll->b_y, 0.0f);

    mat_fill(ll->h_0, 0.0f);
    mat_fill(ll->c_0, 0.0f);
}

void lstm_print(layer l)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    mat_print(ll->w_x_i);
    mat_print(ll->w_x_f);
    mat_print(ll->w_x_o);
    mat_print(ll->w_x_cc);
    mat_print(ll->w_h_i);
    mat_print(ll->w_h_f);
    mat_print(ll->w_h_o);
    mat_print(ll->w_h_cc);
    mat_print(ll->w_y);

    mat_print(ll->b_i);
    mat_print(ll->b_f);
    mat_print(ll->b_o);
    mat_print(ll->b_cc);
    mat_print(ll->b_y);

    mat_print(ll->h_0);
    mat_print(ll->c_0);
}

void lstm_save(layer l, FILE *f)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    mat_save(ll->w_x_i, f);
    mat_save(ll->w_x_f, f);
    mat_save(ll->w_x_o, f);
    mat_save(ll->w_x_cc, f);
    mat_save(ll->w_h_i, f);
    mat_save(ll->w_h_f, f);
    mat_save(ll->w_h_o, f);
    mat_save(ll->w_h_cc, f);
    mat_save(ll->w_y, f);

    mat_save(ll->b_i, f);
    mat_save(ll->b_f, f);
    mat_save(ll->b_o, f);
    mat_save(ll->b_cc, f);
    mat_save(ll->b_y, f);

    mat_save(ll->h_0, f);
    mat_save(ll->c_0, f);
}

void lstm_load(layer l, FILE *f)
{
    lstm_layer *ll = (lstm_layer *)l.data;

    mat_load(ll->w_x_i, f);
    mat_load(ll->w_x_f, f);
    mat_load(ll->w_x_o, f);
    mat_load(ll->w_x_cc, f);
    mat_load(ll->w_h_i, f);
    mat_load(ll->w_h_f, f);
    mat_load(ll->w_h_o, f);
    mat_load(ll->w_h_cc, f);
    mat_load(ll->w_y, f);

    mat_load(ll->b_i, f);
    mat_load(ll->b_f, f);
    mat_load(ll->b_o, f);
    mat_load(ll->b_cc, f);
    mat_load(ll->b_y, f);

    mat_load(ll->h_0, f);
    mat_load(ll->c_0, f);
}
