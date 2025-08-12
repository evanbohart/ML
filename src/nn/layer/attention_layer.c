#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer attention_layer_alloc(int x_size, int e_size, int batch_size,
                            int h_size, int qk_size)
{
    assert(h_size * qk_size == e_size);

    attention_layer *al = malloc(sizeof(attention_layer));

    al->x_size = x_size;
    al->e_size = e_size;
    al->batch_size = batch_size;
    al->h_size = h_size;
    al->qk_size = qk_size;

    al->w_q = tens3D_alloc(e_size, qk_size, h_size);
    al->w_k = tens3D_alloc(e_size, qk_size, h_size);
    al->w_v = tens3D_alloc(e_size, qk_size, h_size);
    al->w_o = mat_alloc(e_size, e_size);

    layer l;

    l.type = ATTENTION;
    l.data = al;
    l.forward = attention_forward;
    l.backprop = attention_backprop;
    l.destroy = attention_destroy;
    l.init = attention_init;
    l.save = attention_save;
    l.load = attention_load;

    return l;
}

void attention_forward(layer l, tens x, tens *y)
{
    attention_layer *al = (attention_layer *)l.data;

    assert(x.type == TENS3D);
    assert(x.t3.rows == al->x_size);
    assert(x.t3.cols == al->e_size);
    assert(x.t3.depth == al->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(al->x_size, al->e_size, al->batch_size);

    tens4D q = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D k = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D v = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < al->h_size; ++i) {
        for (int j = 0; j < al->batch_size; ++j) {
            mat_dot(q.tens3Ds[j].mats[i], x.t3.mats[j], al->w_q.mats[i]);
            mat_dot(k.tens3Ds[j].mats[i], x.t3.mats[j], al->w_k.mats[i]);
            mat_dot(v.tens3Ds[j].mats[i], x.t3.mats[j], al->w_v.mats[i]);
        }
    }

    tens4D_copy(al->q_cache, q);
    tens4D_copy(al->k_cache, k);
    tens4D_copy(al->v_cache, v);

    tens4D k_T = tens4d_alloc(al->qk_size, al->x_size, al->h_size, al->batch_size);
    tens4D_trans(k_T, k);

    tens4D q_k_T  = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D_dot(q_k_T, q, k_T);

    tens4D a = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D_scale(a, q_k_T, 1.0f / sqrtf(al->qk_size));

    tens4D alpha = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D_softmax(alpha, a);
    tens4D_copy(al->alpha_cache, alpha);

    tens4D z = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D_dot(z, alpha, v);

    tens3D concat = tens3D_alloc(al->x_size, al->e_size, al->batch_size);
    tens3D_copy(al->concat_cache, concat);

    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < al->x_size; ++i) {
        for (int j = 0; j < al->qk_size; ++j) {
            for (int k = 0; k < al->h_size; ++k) {
                for (int l = 0; l < al->batch_size; ++l) {
                    tens3D_at(concat, i, k * cl->qk_size + j, l) = tens4D_at(z, i, j, k, l);
                }
            }
        }
    }

    for (int i = 0; i < al->batch_size; ++i) {
        mat_dot(y->t3.mats[i], concat.mats[i], al->w_o);
    }

    tens4D_destroy(q);
    tens4D_destroy(k);
    tens4D_destroy(v);

    tens4D_destroy(k_T);
    tens4D_destroy(q_k_T);
    tens4D_destroy(a);
    tens4D_destroy(alpha);
    tens4D_destroy(z);
    tens3D_destroy(concat);
}

void attention_backprop(layer l, tens dy, tens *dx)
{
    attention_layer *al = (attention_layer *)l.data;

    assert(dy.type == TENS3D);
    assert(dy.t3.rows == al->x_size);
    assert(dy.t3.cols == al->e_size);
    assert(dy.t3.depth == al->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(al->x_size, al->e_size, al->batch_size);

    tens3D x_T = tens3D_alloc(al->x_size, al->e_size, al->batch_size);
    tens3D_trans(x_T, al->x_cache);

    tens3D y_T = tens3D_alloc(al->x_size, al->e_size, al->batch_size);
    tens3D_trans(y_T, al->y_cache);

    tens4D v_T = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D_trans(v_T, al->v_cache);

    tens4D alpha_T = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D_trans(alpha_T, al->alpha_cache);

    tens3D concat_T = tens3D_alloc(al->e_size, al->x_size, al->batch_size);
    tens3D_trans(concat_T, al->concat_cache);

    tens4D z_T = tens4D_alloc(al->qk_size, al->x_size, al->h_size, al->batch_size);
    tens4D_trans(z_T, al->z_cache);

    mat w_o_T = tens3D_alloc(al->e_size, al->e_size);
    mat_trans(w_o_T, al->w_o);

    tens4D dq = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D dk = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens4D dv = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);

    tens4D dalpha = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D da = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);
    tens4D dz = tens4D_alloc(al->x_size, al->qk_size, al->h_size, al->batch_size);
    tens3D dconcat = tens3D_alloc(al->x_size, al->e_size, al->batch_size);

    tens4D da_T = tens4D_alloc(al->x_size, al->x_size, al->h_size, al->batch_size);

    tens3D dw_q_current = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    tens3D dw_k_current = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    tens3D dw_v_current = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    mat dw_o_current = tens3D_alloc(al->e_size, al->e_size);

    tens3D dw_q = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    tens3D_fill(dw_q, 0.0f);

    tens3D dw_k = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    tens3D_fill(dw_k, 0.0f);

    tens3D dw_v = tens3D_alloc(al->e_size, al->qk_size, al->h_size);
    tens3D_fill(dw_v, 0.0f);

    mat dw_o = tens3D_alloc(al->e_size, al->e_size);
    mat_fill(dw_o, 0.0f);

    for (int i = 0; i < al->batch_size; ++i) {
        mat_dot(dconcat.mats[i], dy.t3.mats[i], w_o_T);

        for (int j = 0; j < al->x_size; ++j) {
            for (int k = 0; k < al->qk_size; ++k) {
                for (int l = 0; l < al->h_size; ++l) {
                    tens4D_at(dz, j, k, l, i) = tens3D_at(dconcat, j, l * al->qk_size + k, i);
                }
            }
        }

        tens3D_dot(dalpha.tens3Ds[i], dz.tens3Ds[i], v_T.tens3Ds[i]);

        tens3D_had(da.tens3Ds[i], dalpha.tens3Ds[i], al->alpha_cache.tens3Ds[i]);

        for (int j = 0; j < al->h_size; ++j) {
            for (int k = 0; k < al->x_size; ++k) {
                float sum = 0.0f;

                for (int l = 0; l < al->x_size; ++l) {
                    sum += tens4D_at(dalpha, k, l, j, i) * tens4D_at(al->alpha_cache, k, l, j, i);
                }

                for (int l = 0; l < al->x_size; ++l) {
                    tens4D_at(da, k, l, j, i) -= sum * tens4D_at(al->alpha_cache, k, l, j, i);
                }
            }
        }

        tens3D_dot(dq.tens3Ds[i], da.tens3Ds[i], al->k_cache.tens3Ds[i]);
        tens3D_scale(dq.tens3Ds[i], dq.tens3Ds[i], 1.0f / sqrtf(al->qk_size));

        tens3D_trans(da_trans.tens3Ds[i], da.tens3Ds[i]);

        tens3D_dot(dk.tens3Ds[i], da_trans.tens3Ds[i], al->q_cache.tens3Ds[i]);
        tens3D_scale(dk.tens3Ds[i], dk.tens3Ds[i], 1.0f / sqrtf(al->qk_size));

        tens3D_dot(dw_q_current, x_T, dq.tens3Ds[i]);
        tens3D_dot(dw_k_current, x_T, dk.tens3Ds[i]);
        tens3D_dot(dw_v_current, x_T, dv.tens3Ds[i]);
        mat_dot(dw_o_current, concat_T.mats[i], dy.t3.mats[i]);

        tens3D_add(dw_q, dw_q, dw_q_current);
        tens3D_add(dw_k, dw_k, dw_k_current);
        tens3D_add(dw_v, dw_v, dw_v_current);
        tens3D_add(dw_o, dw_o, dw_o_current);
    }

    tens3D_scale(dw_q, dw_q, rate / al->batch_size);
    tens3D_func(dw_q, dw_q, clip);
    tens3D_sub(al->w_q, al->w_q, dw_q);

    tens3D_scale(dw_k, dw_k, rate / al->batch_size);
    tens3D_func(dw_k, dw_k, clip);
    tens3D_sub(al->w_k, al->w_k, dw_k);

    tens3D_scale(dw_v, dw_v, rate / al->batch_size);
    tens3D_func(dw_v, dw_v, clip);
    tens3D_sub(al->w_v, al->w_v, dw_v);

    tens3D_scale(dw_o, dw_o, rate / al->batch_size);
    tens3D_func(dw_o, dw_o, clip);
    tens3D_sub(al->w_o, al->w_o, dw_o);
 
    tens3D_destroy(x_T);
    tens4D_destroy(v_T);
    tens4D_destroy(alpha_T);
    tens4D_destroy(z_T);
    tens3D_destroy(w_o_T);

    tens4D_destroy(dq);
    tens4D_destroy(dk);
    tens4D_destroy(dv);

    tens4D_destroy(dalpha);
    tens4D_destroy(da);
    tens4D_destroy(dz);

    tens4D_destroy(da_T);

    tens3D_destroy(dw_q_current);
    tens3D_destroy(dw_k_current);
    tens3D_destroy(dw_v_current);
    free(dw_o_current.vals);

    tens3D_destroy(dw_q);
    tens3D_destroy(dw_k);
    tens3D_destroy(dw_v);
    free(dw_o.vals);
}

void attention_destroy(layer l)
{
    attention_layer *al = (attention_layer *)l.data;

    tens3D_destroy(al->w_q);
    tens3D_destroy(al->w_k);
    tens3D_destroy(al->w_v);
    tens3D_destroy(al->w_o);

    free(al);
}

void attention_init(layer l)
{
    attention_layer *al = (attention_layer *)l.data;

    float range = sqrtf(6.0f / (al->e_size + al->qk_size));

    tens3D_rand(al->w_q, -range, range);
    tens3D_rand(al->w_k, -range, range);
    tens3D_tand(al->w_v, -range, range);
    tens3D_rand(al->w_o, -range, range);
}

void attention_print(layer l)
{
    attention_layer *al = (attention_layer *)l.data;

    tens3D_print(al->w_q);
    tens3D_print(al->w_k);
    tens3D_print(al->w_v);
    tens3D_print(al->w_o);
}

void attention_save(layer l, FILE *f)
{
    attention_layer *al = (attention_layer *)l.data;

    tens3D_save(al->w_q, f);
    tens3D_save(al->w_k, f);
    tens3D_save(al->w_v, f);
    tens3D_save(al->w_o, f);
}

void attention_load(layer l, FILE *f)
{
    attentino_layer *al = (attention_layer *)l.data;

    tens3D_load(al->w_q, f);
    tens3D_load(al->w_k, f);
    tens3D_load(al->w_v, f);
    tens3D_load(al->w_o, f);
}
