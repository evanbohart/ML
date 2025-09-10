#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer batchnorm_layer_2D_alloc(int x_size, int batch_size)
{
    batchnorm_layer *bl = malloc(sizeof(batchnorm_layer));

    bl->x_rows = x_size;
    bl->x_cols = batch_size;
    bl->x_depth = 1;
    bl->x_batches = 1;

    bl->gamma = tens2D_alloc(x_size, 1);
    bl->beta = tens2D_alloc(x_size, 1);
    bl->var_cache = tens2D_alloc(x_size, 1);

    bl->z_cache = tens2D_alloc(x_size, batch_size);

    layer l;

    l.data = bl;

    l.forward = batchnorm_2D_forward;
    l.backprop = batchnorm_2D_backprop;
    l.destroy = batchnorm_destroy;

    l.init = batchnorm_init;
    l.print = batchnorm_print;
    l.save = batchnorm_save;
    l.load = batchnorm_load;

    return l;
}

layer batchnorm_layer_4D_alloc(int x_rows, int x_cols,
                               int x_depth, int batch_size)
{
    batchnorm_layer *bl = malloc(sizeof(batchnorm_layer));

    bl->x_rows = x_rows;
    bl->x_cols = x_cols;
    bl->x_depth = x_depth;
    bl->x_batches = batch_size;

    bl->gamma = tens2D_alloc(x_depth, 1);
    bl->beta = tens2D_alloc(x_depth, 1);
    bl->var_cache = tens2D_alloc(x_depth, 1);

    bl->z_cache = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);

    layer l;

    l.data = bl;

    l.forward = batchnorm_4D_forward;
    l.backprop = batchnorm_4D_backprop;
    l.destroy = batchnorm_destroy;

    l.init = batchnorm_init;
    l.print = batchnorm_print;
    l.save = batchnorm_save;
    l.load = batchnorm_load;

    return l;
}

void batchnorm_2D_forward(layer l, tens x, tens *y)
{
}

void batchnorm_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
}

void batchnorm_4D_forward(layer l, tens x, tens *y)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(x.rows == bl->x_rows);
    assert(x.cols == bl->x_cols);
    assert(x.depth == bl->x_depth);
    assert(x.batches == bl->x_batches);

    *y = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->x_batches);

    int n = bl->x_rows * bl->x_cols * bl->x_batches;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        float mean = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->x_batches; ++l) {
                    float x_val = tens4D_at(x, j, k, i, l);
                    mean += x_val;
                }
            }
        }

        mean /= n;

        float var = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->x_batches; ++l) {
                    float x_val = tens4D_at(x, j, k, i, l);
                    float diff = x_val - mean;
                    var += diff * diff;
                }
            }
        }

        var /= n;

        tens2D_at(bl->var_cache, i, 0) = var;

        float eps = 1e-5;
        float stddev = sqrtf(var + eps);

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->x_batches; ++l) {
                    float x_val = tens4D_at(x, j, k, i, l);
                    tens4D_at(bl->z_cache, j, k, i, l) = (x_val - mean) / stddev;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_batches; ++j) {
            for (int k = 0; k < bl->x_rows; ++k) {
                for (int l = 0; l < bl->x_cols; ++l) {
                    float gamma = tens2D_at(bl->gamma, i, 0);
                    float beta = tens2D_at(bl->beta, i, 0);
                    float z = tens4D_at(bl->z_cache, k, l, i, j);
                    tens4D_at(*y, k, l, i, j) = gamma * z + beta;
                }
            }
        }
    }
}

void batchnorm_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(dy.rows == bl->x_rows);
    assert(dy.cols == bl->x_cols);
    assert(dy.depth == bl->x_depth);
    assert(dy.batches == bl->x_batches);

    *dx = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->x_batches);

    tens sum_dy = tens2D_alloc(bl->x_depth, 1);
    tens_fill(sum_dy, 0.0f);

    tens sum_dy_z = tens2D_alloc(bl->x_depth, 1);
    tens_fill(sum_dy_z, 0.0f);

    tens dgamma = tens2D_alloc(bl->x_depth, 1);
    tens_fill(dgamma, 0.0f);

    tens dbeta = tens2D_alloc(bl->x_depth, 1);
    tens_fill(dbeta, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_batches; ++j) {
            for (int k = 0; k < bl->x_rows; ++k) {
                for (int l = 0; l < bl->x_cols; ++l) {
                    float z_val = tens4D_at(bl->z_cache, k, l, i, j);

                    tens2D_at(sum_dy, i, 0) += tens4D_at(dy, k, l, i, j);
                    tens2D_at(sum_dy_z, i, 0) += tens4D_at(dy, k, l, i, j) * z_val;
                }
            }
        }
    }

    int n = bl->x_rows * bl->x_cols * bl->x_batches;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_batches; ++j) {
            for (int k = 0; k < bl->x_rows; ++k) {
                for (int l = 0; l < bl->x_cols; ++l) {
                    float dy_val = tens4D_at(dy, k, l, i, j);
                    float z_val = tens4D_at(bl->z_cache, k, l, i, j);
                    float sum_dy_val = tens2D_at(sum_dy, i, 0);
                    float sum_dy_z_val = tens2D_at(sum_dy_z, i, 0);
 
                    float gamma = tens2D_at(bl->gamma, i, 0);
                    float var = tens2D_at(bl->var_cache, i, 0);
                    float eps = 1e-5;
                    float stddev = sqrtf(var + eps);

                    tens4D_at(*dx, j, k, i, l) =
                        (dy_val - sum_dy_val / n - z_val * sum_dy_z_val / n) * gamma / stddev;
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_batches; ++j) {
            for (int k = 0; k < bl->x_rows; ++k) {
                for (int l = 0; l < bl->x_cols; ++l) {
                    float dy_val = tens4D_at(dy, k, l, i, j);
                    float z = tens4D_at(bl->z_cache, k, l, i, j);

                    tens2D_at(dgamma, i, 0) += dy_val * z;
                    tens2D_at(dbeta, i, 0) += dy_val;
                }
            }
        }
    }

    tens_scale(dgamma, dgamma, rate / bl->x_batches);
    tens_func(dgamma, dgamma, clip);
    tens_sub(bl->gamma, bl->gamma, dgamma);

    tens_scale(dbeta, dbeta, rate / bl->x_batches);
    tens_func(dbeta, dbeta, clip);
    tens_sub(bl->beta, bl->beta, dbeta);

    tens_destroy(sum_dy);
    tens_destroy(sum_dy_z);
    tens_destroy(dgamma);
    tens_destroy(dbeta);
}

void batchnorm_destroy(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    tens_destroy(bl->gamma);
    tens_destroy(bl->beta);
    tens_destroy(bl->var_cache);
    tens_destroy(bl->z_cache);
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
