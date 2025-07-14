#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int batch_size, int filter_size, int y_depth,
                       int stride, padding_t padding, actfunc activation)
{
    conv_layer *cl = malloc(sizeof(conv_layer));

    int y_rows = (x_rows + padding[TOP] + padding[BOTTOM] - filter_size) / stride + 1;
    int y_cols = (x_cols + padding[LEFT] + padding[RIGHT] - filter_size) / stride + 1;

    cl->x_rows = x_rows;
    cl->x_cols = x_cols;
    cl->x_depth = x_depth;
    cl->batch_size = batch_size;
    cl->y_rows = y_rows;
    cl->y_cols = y_cols;
    cl->y_depth = y_depth;
    cl->filter_size = filter_size;
    cl->stride = stride;
    memcpy(cl->padding, padding, sizeof(cl->padding));
    cl->activation = activation;

    cl->filters = tens4D_alloc(filter_size, filter_size, x_depth, y_depth);
    cl->b = tens3D_alloc(y_rows, y_cols, y_depth);

    cl->x_cache = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);
    cl->z_cache = tens4D_alloc(y_rows, y_cols, y_depth, batch_size);

    layer l;
    l.type = CONV;
    l.data = cl;
    l.forward = conv_forward;
    l.backprop = conv_backprop;
    l.destroy = conv_destroy;
    l.init = conv_init;
    l.print = conv_print;
    l.save = conv_save;
    l.load = conv_load;

    return l;
}

void conv_forward(layer l, void *input, void **output)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens4D *tens4D_input = (tens4D *)input;

    assert(tens4D_input->rows == cl->x_rows);
    assert(tens4D_input->cols == cl->x_cols);
    assert(tens4D_input->depth == cl->x_depth);
    assert(tens4D_input->batches == cl->batch_size);

    tens4D *tens4D_output = malloc(sizeof(tens4D));

    tens4D_copy(cl->x_cache, *tens4D_input);

    tens4D x_padded = tens4D_alloc(cl->x_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                        cl->x_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                        cl->x_depth, cl->batch_size);
    tens4D_pad(x_padded, *tens4D_input, cl->padding);

    tens4D_fill(cl->z_cache, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->y_depth; ++j) {
            mat convolved = mat_alloc(cl->y_rows, cl->y_cols);

            for (int k = 0; k < cl->x_depth; ++k) {
                mat_convolve(convolved, x_padded.tens3Ds[i].mats[k],
                             cl->filters.tens3Ds[j].mats[k]);
                mat_add(cl->z_cache.tens3Ds[i].mats[j],
                        cl->z_cache.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->y_depth; ++j) {
            for (int k = 0; k < cl->y_rows; ++k) {
                for (int l = 0; l < cl->y_cols; ++l) {
                    tens4D_at(cl->z_cache, k, l, j, i) += tens3D_at(cl->b, k, l, j);
                }
            }
        }
    }

    *tens4D_output = tens4D_alloc(cl->y_rows, cl->y_cols,
                                    cl->y_depth, cl->batch_size);

    switch (cl->activation) {
        case LIN:
            tens4D_func(*tens4D_output, cl->z_cache, lin);
            break;
        case SIG:
            tens4D_func(*tens4D_output, cl->z_cache, sig);
            break;
        case TANH:
            tens4D_func(*tens4D_output, cl->z_cache, tanhf);
            break;
        case RELU:
            tens4D_func(*tens4D_output, cl->z_cache, relu);
            break;
    }

    *output = tens4D_output;

    tens4D_destroy(x_padded);
}

void conv_backprop(layer l, void *delta_in, void **delta_out, float rate)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens4D *tens4D_delta_in = (tens4D *)delta_in;

    assert(tens4D_delta_in->rows == cl->y_rows);
    assert(tens4D_delta_in->cols == cl->y_cols);
    assert(tens4D_delta_in->depth == cl->y_depth);
    assert(tens4D_delta_in->batches == cl->batch_size);

    tens4D *tens4D_delta_out = malloc(sizeof(tens4D));
    tens4D dz = tens4D_alloc(cl->y_rows, cl->y_cols,
                                     cl->y_depth, cl->batch_size);

    switch (cl->activation) {
        case LIN:
            tens4D_func(dz, cl->z_cache, dlin);
            break;
        case SIG:
            tens4D_func(dz, cl->z_cache, dsig);
            break;
        case TANH:
            tens4D_func(dz, cl->z_cache, dtanh);
            break;
        case RELU:
            tens4D_func(dz, cl->z_cache, drelu);
            break;
    }

    tens4D delta = tens4D_alloc(cl->y_rows, cl->y_cols,
                               cl->y_depth, cl->batch_size);
    tens4D_had(delta, *tens4D_delta_in, dz);

    padding_t delta_padding = { cl->filter_size - 1 - cl->padding[TOP],
                               cl->filter_size - 1 - cl->padding[BOTTOM],
                               cl->filter_size - 1 - cl->padding[LEFT],
                               cl->filter_size - 1 - cl->padding[RIGHT] };

    tens4D delta_padded = tens4D_alloc(cl->y_rows + delta_padding[TOP] + delta_padding[BOTTOM],
                                      cl->y_cols + delta_padding[LEFT] + delta_padding[RIGHT],
                                      cl->y_depth, cl->batch_size);
    tens4D_pad(delta_padded, delta, delta_padding);

    tens4D filter_180 = tens4D_alloc(cl->filter_size, cl->filter_size,
                                     cl->x_depth, cl->y_depth);

    tens4D_180(filter_180, cl->filters);

    *tens4D_delta_out = tens4D_alloc(cl->x_rows, cl->x_cols,
                                    cl->x_depth, cl->batch_size);
    tens4D_fill(*tens4D_delta_out, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->x_depth; ++j) {
            mat convolved = mat_alloc(cl->x_rows, cl->x_cols);

            for (int k = 0; k < cl->y_depth; ++k) {
                mat_convolve(convolved, delta_padded.tens3Ds[i].mats[k],
                             filter_180.tens3Ds[k].mats[j]);
                mat_add(tens4D_delta_out->tens3Ds[i].mats[j],
                        tens4D_delta_out->tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    *delta_out = tens4D_delta_out;

    tens4D x_padded = tens4D_alloc(cl->x_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                        cl->x_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                        cl->x_depth, cl->batch_size);
    tens4D_pad(x_padded, cl->x_cache, cl->padding);

    tens4D dw = tens4D_alloc(cl->filter_size, cl->filter_size,
                             cl->x_depth, cl->y_depth);
    tens4D_fill(dw, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->y_depth; ++i) {
        for (int j = 0; j < cl->x_depth; ++j) {
            mat convolved = mat_alloc(cl->filter_size, cl->filter_size);

            for (int k = 0; k < cl->batch_size; ++k) {
                mat_convolve(convolved, x_padded.tens3Ds[k].mats[j],
                             delta.tens3Ds[k].mats[i]);
                mat_add(dw.tens3Ds[i].mats[j], dw.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    tens4D_scale(dw, dw, 1.0 / cl->batch_size);
    tens4D_func(dw, dw, clip);
    tens4D_scale(dw, dw, rate);
    tens4D_sub(cl->filters, cl->filters, dw);

    tens3D db = tens3D_alloc(cl->y_rows, cl->y_cols, cl->y_depth);
    tens3D_fill(db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cl->y_depth; ++i) {
        for (int j = 0; j < cl->y_rows; ++j) {
            for (int k = 0; k < cl->y_cols; ++k) {
                float sum = 0;

                for (int l = 0; l < cl->batch_size; ++l) {
                    sum += tens4D_at(delta, j, k, i, l);
                }

                tens3D_at(db, j, k, i) = sum;
            }
        }
    }

    tens3D_scale(db, db, 1.0 / cl->batch_size);
    tens3D_func(db, db, clip);
    tens3D_scale(db, db, rate);
    tens3D_sub(cl->b, cl->b, db);

    tens4D_destroy(dz);
    tens4D_destroy(delta);
    tens4D_destroy(delta_padded);
    tens4D_destroy(filter_180);
    tens4D_destroy(x_padded);
    tens4D_destroy(dw);
    tens3D_destroy(db);
}

void conv_destroy(layer l)
{
   conv_layer *cl = (conv_layer *)l.data;

    tens4D_destroy(cl->filters);
    tens3D_destroy(cl->b);
    tens4D_destroy(cl->x_cache);
    tens4D_destroy(cl->z_cache);

    free(cl);
}

void conv_init(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    if (cl->activation == SIG || cl->activation == TANH) {
        tens4D_normal(cl->filters, 0, sqrt(2.0 / (cl->x_depth * cl->filter_size * cl->filter_size +
                                                  cl->y_depth * cl->filter_size * cl->filter_size)));
    }
    else {
        tens4D_normal(cl->filters, 0, sqrt(2.0 / (cl->x_depth * cl->filter_size * cl->filter_size)));
    }

    tens3D_fill(cl->b, 0);
}

void conv_print(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_print(cl->filters);
    tens3D_print(cl->b);
}

void conv_save(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_save(cl->filters, f);
    tens3D_save(cl->b, f);
}

void conv_load(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_load(cl->filters, f);
    tens3D_load(cl->b, f);
}
