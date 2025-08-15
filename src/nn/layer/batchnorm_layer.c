#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer batchnorm_layer_2D_alloc(int x_size, int batch_size)
{
    batchnorm_layer *bl = malloc(sizeof(batchnorm_layer));

    bl->x_type = MAT;
    bl->x_rows = x_size;
    bl->batch_size = batch_size;

    bl->gamma = mat_alloc(x_size, 1);
    bl->beta = mat_alloc(x_size, 1);
    bl->var_cache = mat_alloc(x_size, 1);

    bl->z_cache.type = MAT;
    bl->z_cache.m = mat_alloc(x_size, batch_size);

    layer l;

    l.data = bl;

    l.forward = batchnorm_2D_forward;
    l.backprop = batchnorm_2D_backprop;
    l.destroy = batchnorm_2D_destroy;

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

    bl->x_type = TENS4D;
    bl->x_rows = x_rows;
    bl->x_cols = x_cols;
    bl->x_depth = x_depth;
    bl->batch_size = batch_size;

    bl->gamma = mat_alloc(x_depth, 1);
    bl->beta = mat_alloc(x_depth, 1);
    bl->var_cache = mat_alloc(x_depth, 1);

    bl->z_cache.type = TENS4D;
    bl->z_cache.t4 = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);


    layer l;

    l.data = bl;

    l.forward = batchnorm_4D_forward;
    l.backprop = batchnorm_4D_backprop;
    l.destroy = batchnorm_4D_destroy;

    l.init = batchnorm_init;
    l.print = batchnorm_print;
    l.save = batchnorm_save;
    l.load = batchnorm_load;

    return l;
}


void batchnorm_2D_forward(layer l, tens x, tens *y)
{
/**    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(x.type == MAT);
    assert(bl->x_type == MAT);

    assert(x.m.rows == bl->x_rows);
    assert(x.m.cols == bl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(bl->x_rows, bl->batch_size);

    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->batch_size; ++j) {
            float gamma = mat_at(bl->gamma, i, 0);
            float beta = mat_at(bl->beta, i, 0);
            float z = mat_at(bl->z_cache.m, i, j);

            mat_at(y->m, i, j) = gamma * z + beta;
        }
    } **/
}

void batchnorm_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
}

void batchnorm_2D_destroy(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(bl->x_type == MAT);

    free(bl->gamma.vals);
    free(bl->beta.vals);
    free(bl->var_cache.vals);
    free(bl->z_cache.m.vals);

}

void batchnorm_4D_forward(layer l, tens x, tens *y)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(bl->x_type == TENS4D);

    assert(x.type == TENS4D);
    assert(x.t4.rows == bl->x_rows);
    assert(x.t4.cols == bl->x_cols);
    assert(x.t4.depth == bl->x_depth);
    assert(x.t4.batches == bl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->batch_size);

    int n = bl->x_rows * bl->x_cols * bl->batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        float mean = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float x_val = tens4D_at(x.t4, j, k, i, l);
                    mean += x_val;
                }
            }
        }

        mean /= n;

        float var = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float x_val = tens4D_at(x.t4, j, k, i, l);
                    float diff = x_val - mean;
                    var += diff * diff;
                }
            }
        }

        var /= n;

        mat_at(bl->var_cache, i, 0) = var;

        float eps = 1e-5;
        float stddev = sqrtf(var + eps);

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float x_val = tens4D_at(x.t4, j, k, i, l);
                    tens4D_at(bl->z_cache.t4, j, k, i, l) = (x_val - mean) / stddev;
                }
            }
        }
    }

    #pragma omp parallel for schedule(static) collapse(4)
    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->x_cols; ++j) {
            for (int k = 0; k < bl->x_depth; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float gamma = mat_at(bl->gamma, k, 0);
                    float beta = mat_at(bl->beta, k, 0);
                    float z = tens4D_at(bl->z_cache.t4, i, j, k, l);
                    tens4D_at(y->t4, i, j, k, l) = gamma * z + beta;
                }
            }
        }
    }
}

void batchnorm_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(bl->x_type == TENS4D);

    assert(dy.type == TENS4D);
    assert(dy.t4.rows == bl->x_rows);
    assert(dy.t4.cols == bl->x_cols);
    assert(dy.t4.depth == bl->x_depth);
    assert(dy.t4.batches == bl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->batch_size);

    mat sum_dy = mat_alloc(bl->x_depth, 1);
    mat_fill(sum_dy, 0.0f);

    mat sum_dy_z = mat_alloc(bl->x_depth, 1);
    mat_fill(sum_dy_z, 0.0f);

    mat dgamma = mat_alloc(bl->x_depth, 1);
    mat_fill(dgamma, 0.0f);

    mat dbeta = mat_alloc(bl->x_depth, 1);
    mat_fill(dbeta, 0.0f);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float z = tens4D_at(bl->z_cache.t4, j, k, i, l);

                    mat_at(sum_dy, i, 0) += tens4D_at(dy.t4, j, k, i, l);
                    mat_at(sum_dy_z, i, 0) += tens4D_at(dy.t4, j, k, i, l) * z;
                }
            }
        }
    }

    int n = bl->x_rows * bl->x_cols * bl->batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < bl->x_depth; ++i) {
        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float dy_val = tens4D_at(dy.t4, j, k, i, l);
                    float z = tens4D_at(bl->z_cache.t4, j, k, i, l);
 
                    float gamma = mat_at(bl->gamma, i, 0);
                    float var = mat_at(bl->var_cache, i, 0);
                    float eps = 1e-5;
                    float stddev = sqrtf(var + eps);

                    tens4D_at(dx->t4, j, k, i, l) =
                        (dy_val - mat_at(sum_dy, i, 0) / n - z * mat_at(sum_dy_z, i, 0) / n) * gamma / stddev;

                    mat_at(dgamma, i, 0) += dy_val * z;
                    mat_at(dbeta, i, 0) += dy_val;
                }
            }
        }
    }

    mat_scale(dgamma, dgamma, rate / bl->batch_size);
    mat_func(dgamma, dgamma, clip);
    mat_sub(bl->gamma, bl->gamma, dgamma);

    mat_scale(dbeta, dbeta, rate / bl->batch_size);
    mat_func(dbeta, dbeta, clip);
    mat_sub(bl->beta, bl->beta, dbeta);

    free(sum_dy.vals);
    free(sum_dy_z.vals);
    free(dgamma.vals);
    free(dbeta.vals);
}

void batchnorm_4D_destroy(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    free(bl->gamma.vals);
    free(bl->beta.vals);
    free(bl->var_cache.vals);
    tens4D_destroy(bl->z_cache.t4);
}

void batchnorm_init(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    mat_fill(bl->gamma, 1.0f);
    mat_fill(bl->beta, 0.0f);
}

void batchnorm_print(layer l)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    mat_print(bl->gamma);
    mat_print(bl->beta);
}

void batchnorm_save(layer l, FILE *f)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    mat_save(bl->gamma, f);
    mat_save(bl->beta, f);
}

void batchnorm_load(layer l, FILE *f)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    mat_load(bl->gamma, f);
    mat_load(bl->beta, f);
}
