#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

layer layernorm_2D_alloc(int x_size, int batch_size)
{
    layernorm_layer *ll = malloc(sizeof(layernorm_layer));

    ll->x_rows = x_size;
    ll->x_cols = batch_size;
    ll->x_depth = 1;
    ll->x_batches = 1;

    ll->gamma = tens2D_alloc(x_size, 1);
    ll->beta = tens2D_alloc(x_size, 1);
    ll->var_cache = tens2D_alloc(batch_size, 1);

    ll->z_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = ll;

    l.forward = layernorm_2D_forward;
    l.backprop = layernorm_2D_backprop;
    l.destroy = layernorm_2D_destroy;

    l.init = layernorm_init;
    l.print = layernorm_print;
    l.save = layernorm_save;
    l.load = layernorm_load;

    return l;
}

layer layernorm_layer_3D_alloc(int x_rows, int x_cols, int batch_size)
{
    layernorm_layer *ll = malloc(sizeof(layernorm_layer));

    ll->x_rows = x_rows;
    ll->x_cols = x_cols;
    ll->x_depth = batch_size;

    ll->gamma = tens2D_alloc(x_rows, 1);
    ll->beta = tens2D_alloc(x_rows, 1);
    ll->var_cache = tens2D_alloc(batch_size, x_cols);

    ll->z_cache.type = TENS3D;
    ll->z_cache.t3 = tens3D_alloc(x_rows, x_cols, batch_size);

    layer l;

    l.data = ll;

    l.forward = layernorm_3D_forward;
    l.backprop = layernorm_3D_backprop;
    l.destroy = layernorm_3D_destroy;

    l.init = layernorm_init;
    l.print = layernorm_print;
    l.save = layernorm_save;
    l.load = layernorm_load;
}

void layernorm_2D_forward(layer l, tens x, tens *y)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == MAT);

    assert(x.type == MAT);
    assert(x.m.rows == ll->x_rows);
    assert(x.m.cols == ll->batch_size);

    y->type = MAT;
    y->m = tens2D_alloc(ll->x_rows, ll->batch_size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->batch_size; ++i) {
        float mean = 0.0f;

        for (int j = 0; j < ll->x_rows; ++j) {
            float x_val = mat_at(x.m, j, i);
            mean += x_val;
        }

        mean /= ll->x_rows;

        float var = 0.0f;

        for (int j = 0; j < ll->x_rows; ++j) {
            float x_val = mat_at(x.m, j, i);
            float diff = x_val - mean;
            var += diff * diff;
        }

        var /= ll->x_rows;

        mat_at(ll->var_cache, i, 0) = var;

        float eps = 1e-5;
        float stddev = sqrtf(var + eps);

        for (int j = 0; j < ll->x_rows; ++j) {
            float x_val = mat_at(x.m, j, i);
            mat_at(ll->z_cache.m, j, i) = (x_val - mean) / stddev;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ll->x_rows; ++i) {
        for (int j = 0; j < ll->batch_size; ++j) {
            float z_val = mat_at(ll->z_cache.m, i, j);
            float gamma = mat_at(ll->gamma, i, 0);
            float beta = mat_at(ll->beta, i, 0);
            mat_at(y->m, i, j) = gamma * z_val + beta;
        }
    }
}

void layernorm_2D_backprop(layer l, tens dy, tens *dx)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == MAT);

    assert(dy.type == MAT);
    assert(dy.m.rows == ll->x_rows);
    assert(dy.m.cols == ll->batch_size);

    dx->type = MAT;
    dx->m = tens2D_alloc(ll->x_rows, ll->batch_size);

    mat sum_dy = tens2D_alloc(ll->batch_size, 1);
    mat sum_dy_z = tens2D_alloc(ll->batch_size, 1);

    mat dgamma = tens2D_alloc(ll->x_rows, 1);
    mat_fill(dgamma, 0.0f);

    mat dbeta = tens2D_alloc(ll->x_rows, 1);
    mat_fill(dbeta, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->batch_size; ++i) {
        for (int j = 0; j < ll->x_size; ++j) {
            float dy_val = mat_at(dy.m, j, i);
            float z_val = mat_at(ll->z_cache.m, i, j);

            mat_at(sum_dy, i, 0) += dy_val;
            mat_at(sum_dy_z, i, 0) += dy_val * z_val;
        }
    }

    for (int i = 0; i < ll->batch_size; ++i) {
        for (int j = 0; j < ll->x_size; ++j) {
            float dy_val = mat_at(dy.m, j, i);
            float z_val = mat_at(ll->z_cache.m, j, i);
            float sum_dy_val = mat_at(sum_dy, i, 0);
            float sum_dy_z_val = mat_at(sum_dy_z, i, 0);

            float gamma = mat_at(ll->gamma, j, 0);
            float var = mat_at(ll->var_cache, i, 0);
            float eps = 1e-5;
            float stddev = sqrtf(var + eps);

            mat_at(dx.m, j, i) =
                (dy_val - sum_dy_val / ll->x_rows - z_val * sum_dy_z_val / ll->x_rows) * gamma / stddev;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->x_rows; ++i) {
        for (int j = 0; j < ll->batch_size; ++j) {
            float dy_val = mat_at(dy.m, i, j);
            float z_val = mat_at(ll->z_cache.m, i, j);

            mat_at(dgamma, i, 0) += dy_val * z_val;
            mat_at(dbeta, i, 0) += dy_val;
        }
    }
}

void layernorm_2D_destroy(layer l)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == MAT);

    free(ll->gamma.vals);
    free(ll->beta.bals);
    free(ll->var_cache.vals);
    free(ll->z_cache.m.vals);

    free(ll);
}

void layernorm_3D_forward(layer l, tens x, tens *y)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == TENS3D);

    assert(x.type == TENS3D);
    assert(x.t3.rows == ll->x_rows);
    assert(x.t3.cols == ll->x_cols);
    assert(x.t3.depth == ll->batch_size);

    y->type = TENS3D;
    y->t3 = tens3D_alloc(ll->x_rows, ll->x_cols, ll->batch_size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->batch_size; ++i) {
        for (int j = 0; j < ll->x_cols; ++j) {
            float mean = 0.0f;

            for (int k = 0; k < ll->x_rows; ++k) {
                float x_val = tens3D_at(x.t3, k, j, i);
                mean += x_val;
            }

            mean /= ll->x_rows;

            float var = 0.0f;

            for (int k = 0; k < ll->x_rows; ++k) {
                float x_val = tens3D_at(x.t3, k, j, i);
                float diff = x_val - mean;
                var += diff * diff;
            }

            var /= ll->x_rows;

            mat_at(ll->var_cache, j, i) = var;

            float eps = 1e-5;
            float stddev = sqrtf(var + eps);

            for (int k = 0; k < ll->x_rows; ++k) {
                float x_val = tens3D_at(x.t3, k, j, i);
                tens3D_at(ll->z_cache.t3, k, j, i) = (x_val - mean) / stddev;
            }
        }
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < ll->x_rows; ++i) {
        for (int j = 0; j < ll->x_cols; ++j) {
            for (int k = 0; k < ll->batch_size; ++k) {
                float z_val = mat_at(l->z_cache.t3, i, j, k);
                float gamma = mat_at(ll->gamma, i, 0);
                float beta = mat_at(ll->beta, i, 0);

                mat_at(y->t3, i, j, k) = gamma * z_val + beta;
            }
        }
    }
}

void layernorm_3D_backprop(layer l, tens dy, tens *dx, float rate)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == TENS3D);

    assert(dy.type == TENS3D);
    assert(dy.t3.rows == ll->x_rows);
    assert(dy.t3.cols == ll->x_cols);
    assert(dy.t3.depth == ll->batch_size);

    dx->type = TENS3D;
    dx->t3 = tens3D_alloc(ll->x_rows, ll->x_cols, ll->batch_size);

    mat dgamma = tens2D_alloc(ll->x_rows, 1);
    mat dbeta = tens2D_alloc(ll->x_rows, 1);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->batch_size; ++i) {
        for (int j = 0; j < ll->x_cols; ++j) {
            for (int k = 0; k < ll->x_rows; ++k) {
                float dy_val = tens3D_at(dy.t3, k, j, i);
                float z_val = tens3D_at(ll->z_cache.t3, k, j, i);

                mat_at(sum_dy, i, 0) += dy_val;
                mat_at(sum_dy_z, i, 0) += dy_val * z_val;
            }
        }
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < ll->batch_size; ++i) {
        for (int j = 0; j < ll->x_cols; ++j) {
            for (int k = 0; k < ll->x_rows; ++k) {
                float dy_val = mat_at(dy.t3, k, j, i)
                float z_val = mat_at(ll->z_cache.t3, k, j, i);
                float sum_dy_val = mat_at(sum_dy, i, 0);
                float sum_dy_z_val = mat_at(sum_dy_z, i, 0);

                float gamma = mat_at(ll->gamma, k, 0);
                float var = mat_at(ll->var_cache, j, i);
                float eps = 1e-5;
                float stddev = sqrtf(var + eps);

                mat_at(dx.t3, k j, i) =
                    (dy_val - sum_dy_val / ll->x_rows - z_val * sum_dy_z_val / ll->x_rows) * gamma / stddev;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ll->x_rows; ++i) {
        for (int j = 0; j < ll->x_cols; ++j) {
            for (int k = 0; k < ll->batch_size; ++k) {
                float dy_val = tens3D_at(dy.t3, i, j, k);
                float z_val = tens3D_at(ll->z_cache.t3, i, j, k);

                mat_at(dgamma, i, 0) += dy_val * z_val;
                mat_at(dbeta, i, 0) += dy_val;
            }
        }
    }
}

void layernorm_3D_destroy(layer l)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    assert(ll->x_type == TENS3D);

    free(ll->gamma.vals);
    free(ll->beta.vals);
    free(ll->var_cache.vals);
    tens3D_destroy(ll->z_cache.t3);

    free(ll);
}

void layernorm_init(layer l)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    mat_fill(ll->gamma, 1.0f);
    mat_fill(ll->beta, 0.0f);
}

void layernorm_print(layer l)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    mat_print(ll->gamma);
    mat_print(ll->bata);
}

void layernorm_save(layer l, FILE *f)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    mat_save(ll->gamma, f);
    mat_save(ll->bata, f);
}

void layernorm_load(layer l, FILE *f)
{
    layernorm_layer *ll = (layernorm_layer *)l.data;

    mat_load(ll->gamma, f);
    mat_load(ll->bata, f);
}
