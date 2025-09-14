#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer batchnorm_layer_alloc(int x_r, int x_c,
                            int x_d, int batch_size)
{
    batchnorm_layer *bl = malloc(sizeof(batchnorm_layer));

    bl->x_r = x_r;
    bl->x_c = x_c;
    bl->x_d = x_d;
    bl->x_b = batch_size;

    bl->gamma = tens_alloc(x_d, 1, 1, 1);
    bl->beta = tens_alloc(x_d, 1, 1, 1);

    bl->var_cache = tens_alloc(x_d, 1, 1, 1);
    bl->z_cache = tens_alloc(x_r, x_c, x_d, batch_size);

    bl->dgamma = tens_alloc(bl->x_d, 1, 1, 1);
    bl->dbeta = tens_alloc(bl->x_d, 1, 1, 1);

    layer l;

    l.data = bl;

    l.forward = batchnorm_forward;
    l.backprop = batchnorm_backprop;
    l.destroy = batchnorm_destroy;

    l.init = batchnorm_init;
    l.print = batchnorm_print;
    l.save = batchnorm_save;
    l.load = batchnorm_load;

    return l;
}

void batchnorm_forward(layer l, tens x, tens *y)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(x.dims[R] == bl->x_r);
    assert(x.dims[C] == bl->x_c);
    assert(x.dims[D] == bl->x_d);
    assert(x.dims[B] == bl->x_b);

    *y = tens_alloc(bl->x_r, bl->x_c, bl->x_d, bl->x_b);

    int n = bl->x_r * bl->x_c * bl->x_b;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_d; ++i) {
        float mean = 0.0f;

        for (int j = 0; j < bl->x_r; ++j) {
            for (int k = 0; k < bl->x_c; ++k) {
                for (int l = 0; l < bl->x_b; ++l) {
                    float x_val = tens_at(x, j, k, i, l);
                    mean += x_val;
                }
            }
        }

        mean /= n;

        float var = 0.0f;

        for (int j = 0; j < bl->x_r; ++j) {
            for (int k = 0; k < bl->x_c; ++k) {
                for (int l = 0; l < bl->x_b; ++l) {
                    float x_val = tens_at(x, j, k, i, l);
                    float diff = x_val - mean;
                    var += diff * diff;
                }
            }
        }

        var /= n;

        tens_at(bl->var_cache, i, 0, 0, 0) = var;

        float eps = 1e-5;
        float stddev = sqrtf(var + eps);

        for (int j = 0; j < bl->x_r; ++j) {
            for (int k = 0; k < bl->x_c; ++k) {
                for (int l = 0; l < bl->x_b; ++l) {
                    float x_val = tens_at(x, j, k, i, l);
                    tens_at(bl->z_cache, j, k, i, l) = (x_val - mean) / stddev;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < bl->x_d; ++i) {
        for (int j = 0; j < bl->x_b; ++j) {
            for (int k = 0; k < bl->x_r; ++k) {
                for (int l = 0; l < bl->x_c; ++l) {
                    float gamma = tens_at(bl->gamma, i, 0, 0, 0);
                    float beta = tens_at(bl->beta, i, 0, 0, 0);
                    float z = tens_at(bl->z_cache, k, l, i, j);
                    tens_at(*y, k, l, i, j) = gamma * z + beta;
                }
            }
        }
    }
}

void batchnorm_backprop(layer l, tens dy, tens *dx, float rate)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(dy.dims[R] == bl->x_r);
    assert(dy.dims[C] == bl->x_c);
    assert(dy.dims[D] == bl->x_d);
    assert(dy.dims[B] == bl->x_b);

    *dx = tens_alloc(bl->x_r, bl->x_c, bl->x_d, bl->x_b);
 
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_d; ++i) {
        float dgamma_sum = 0.0f;
        float dbeta_sum = 0.0f;

        for (int j = 0; j < bl->x_b; ++j) {
            for (int k = 0; k < bl->x_r; ++k) {
                for (int l = 0; l < bl->x_c; ++l) {
                    float dy_val = tens_at(dy, k, l, i, j);
                    float z_val = tens_at(bl->z_cache, k, l, i, j);

                    dgamma_sum += dy_val * z_val;
                    dbeta_sum += dy_val;
                }
            }
        }

        tens_at(bl->dgamma, i, 0, 0, 0) = dgamma_sum;
        tens_at(bl->dbeta, i, 0, 0, 0) = dbeta_sum;
    }

    int n = bl->x_r * bl->x_c * bl->x_b;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < bl->x_d; ++i) {
        for (int j = 0; j < bl->x_b; ++j) {
            for (int k = 0; k < bl->x_r; ++k) {
                for (int l = 0; l < bl->x_c; ++l) {
                    float dy_val = tens_at(dy, k, l, i, j);
                    float z_val = tens_at(bl->z_cache, k, l, i, j);
                    float dgamma_val = tens_at(bl->dgamma, i, 0, 0, 0);
                    float dbeta_val = tens_at(bl->dbeta, i, 0, 0, 0);
 
                    float gamma = tens_at(bl->gamma, i, 0, 0, 0);
                    float var = tens_at(bl->var_cache, i, 0, 0, 0);
                    float eps = 1e-5;
                    float stddev = sqrtf(var + eps);

                    tens_at(*dx, j, k, i, l) =
                        (dy_val - dbeta_val / n - z_val * dgamma_val / n) * gamma / stddev;
                }
            }
        }
    }

    tens_scale(bl->dgamma, bl->dgamma, rate / bl->x_b);
    tens_func(bl->dgamma, bl->dgamma, clip);
    tens_sub(bl->gamma, bl->gamma, bl->dgamma);

    tens_scale(bl->dbeta, bl->dbeta, rate / bl->x_b);
    tens_func(bl->dbeta, bl->dbeta, clip);
    tens_sub(bl->beta, bl->beta, bl->dbeta);
}

void batchnorm_destroy(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_destroy(bl->gamma);
    tens_destroy(bl->beta);

    tens_destroy(bl->var_cache);
    tens_destroy(bl->z_cache);

    tens_destroy(bl->dgamma);
    tens_destroy(bl->dbeta);
}

void batchnorm_init(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_fill(bl->gamma, 1.0f);
    tens_fill(bl->beta, 0.0f);
}

void batchnorm_print(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_print(bl->gamma);
    tens_print(bl->beta);
}

void batchnorm_save(layer l, FILE *f)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_save(bl->gamma, f);
    tens_save(bl->beta, f);
}

void batchnorm_load(layer l, FILE *f)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_load(bl->gamma, f);
    tens_load(bl->beta, f);
}
