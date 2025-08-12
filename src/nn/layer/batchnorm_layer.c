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

    l.type = BATCHNORM;
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

    l.type = BATCHNORM;
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
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(x.type == MAT);
    assert(bl->x_type == MAT);

    assert(x.m.rows == bl->x_rows);
    assert(x.m.cols == bl->batch_size);

    y->type = MAT;
    y->m = mat_alloc(bl->x_rows, bl->batch_size);

    mat_batchnorm(y->m, x.m);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->batch_size; ++j) {
            mat_at(y->m, i, j) *= mat_at(bl->gamma, i, 0);
            mat_at(y->m, i, j) += mat_at(bl->beta, i, 0);
        }
    }
}

void batchnorm_4D_forward(layer l, tens x, tens *y)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(x.type = TENS4D);
    assert(bl->x_type = TENS4D);

    assert(x.t4.rows == bl->x_rows);
    assert(x.t4.cols == bl->x_cols);
    assert(x.t4.depth == bl->x_depth);
    assert(x.t4.batches == bl->batch_size);

    y->type = TENS4D;
    t->t4 = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->batch_size);

    int n = bl->x_rows * bl->x_cols * bl->batch_size;

    for (int i = 0; i < bl->x_depth; ++i) {
        float x_val = tens4D-at(x.t4, j, k, i, l);

        float mean = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    mean += x_val;
                }
            }
        }

        mean /= n;

        float var = 0.0f;

        for (int j = 0; j < bl->x_rows; ++j) {
            for (int k = 0; k < bl->x_cols; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
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
                    tens4D_at(bl->z_cache j, k, i, l) = (x_val - mean) / stddev;
                }
            }
        }
    }

    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->x_cols; ++j) {
            for (int k = 0; k < bl->x_depth; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float gamma = mat_at(bl->gamma, k, 0);
                    float beta = mat_at(bl->beta, k, 0);
                    float z = tens4D_at(bl->z_cache, i, j, k, l);
                    tens4D_at(y->t4, i, j, k, l) = gamma * z + beta;
                }
            }
        }
    }
}

void batchnorm_2D_backprop(layer l, tens dy, tens *dx, float rate)
{
}

void batchnorm_4D_backprop(layer l, tens dy, tens *dx, float rate)
{
    batchnorm_layer *bl = (batchnorm_layer *)l.data;

    assert(bl->type == TENS4D);

    assert(dy.type == TENS4D);
    assert(dy.rows == bl->x_rows);
    assert(dy.cols == bl->x_cols);
    assert(dy.depth == bl->x_depth);
    assert(dy.batches == bl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(bl->x_rows, bl->x_cols, bl->x_depth, bl->batch_size);

    mat sum_dy = mat_alloc(bl->x_depth, 1);
    mat_fill(sum_dy, 0.0f);

    mat sum_dz_z = mat_alloc(bl->x_depth, 1);
    mat_fill(sum_dz_z, 0.0f);

    mat dgamma = mat_alloc(bl->x_depth, 1);
    mat_fill(dgamma, 0.0f);

    mat dbeta = mat_alloc(bl->x_depth, 1);
    mat_fill(dbeta, 0.0f);

    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->x_cols; ++j) {
            for (int k = 0; k < bl->x_depth; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float z = bl->z_cache(i, j, k, l);

                    mat_at(sum_dy, k, 0) += tens4D_at(dy.t4, i, j, k, l);
                    mat_at(sum_dz_z, k, 0) += tens4D_at(dy.t4, i, j, k, l) * z;
                }
            }
        }
    }

    int n = bl->x_rows * bl->x_cols * bl->batch_size;

    for (int i = 0; i < bl->x_rows; ++i) {
        for (int j = 0; j < bl->x_cols; ++j) {
            for (int k = 0; k < bl->x_depth; ++k) {
                for (int l = 0; l < bl->batch_size; ++l) {
                    float dy_val = tens4D-at(dy->t4, i, j, k, l);
                    float z = bl->z_cache(i, j, k, l);
 
                    float gamma = mat_at(bl->gamma, k, 0);
                    float var = mat_at(bl->var_cache, k, 0);
                    float eps = 1e-5;
                    float stddev = sqrtf(var + eps);

                    tens4D_at(dx->t4, i, j, k, l) =
                        (dy_val - mat_at(sum_dy, k, 0) / n - z * sum_dz_z / n) * gamma / stddev;

                    mat_at(dgamma, k, 0) += dy_val * z;
                    mat_at(dbeta, k, 0) += dy_val;
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

    free(sum_dz.vals);
    free(sum_dz_z.vals);
    free(dgamma.vals);
    free(dbeta.vals);
}
