#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int batch_size, int convolutions,
                       int filter_size, int stride, padding_t padding)
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
    cl->convolutions = convolutions;
    cl->filter_size = filter_size;
    cl->stride = stride;
    memcpy(cl->padding, padding, sizeof(cl->padding));

    cl->filters = tens4D_alloc(filter_size, filter_size, x_depth, convolutions);
    cl->b = tens3D_alloc(y_rows, y_cols, convolutions);

    cl->x_cache = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);

    layer l;
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

void conv_forward(layer l, tens x, tens *y)
{
    conv_layer *cl = (conv_layer *)l.data;

    assert(x.type == TENS4D);
    assert(x.t4.rows == cl->x_rows);
    assert(x.t4.cols == cl->x_cols);
    assert(x.t4.depth == cl->x_depth);
    assert(x.t4.batches == cl->batch_size);

    y->type = TENS4D;
    y->t4 = tens4D_alloc(cl->y_rows, cl->y_cols,
                         cl->convolutions, cl->batch_size);

    tens4D_copy(cl->x_cache, x.t4);

    tens4D x_padded = tens4D_alloc(cl->x_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                   cl->x_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                   cl->x_depth, cl->batch_size);
    tens4D_pad(x_padded, x.t4, cl->padding);

    tens4D_fill(y->t4, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            mat convolved = mat_alloc(cl->y_rows, cl->y_cols);

            for (int k = 0; k < cl->x_depth; ++k) {
                mat_convolve(convolved, x_padded.tens3Ds[i].mats[k],
                             cl->filters.tens3Ds[j].mats[k]);
                mat_add(y->t4.tens3Ds[i].mats[j],
                        y->t4.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    #pragma omp parallel for collapse(4) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            for (int k = 0; k < cl->y_rows; ++k) {
                for (int l = 0; l < cl->y_cols; ++l) {
                    tens4D_at(y->t4, k, l, j, i) += tens3D_at(cl->b, k, l, j);
                }
            }
        }
    }

    tens4D_destroy(x_padded);
}

void conv_backprop(layer l, tens dy, tens *dx, float rate)
{
    conv_layer *cl = (conv_layer *)l.data;

    assert(dy.type == TENS4D);
    assert(dy.t4.rows == cl->y_rows);
    assert(dy.t4.cols == cl->y_cols);
    assert(dy.t4.depth == cl->convolutions);
    assert(dy.t4.batches == cl->batch_size);

    dx->type = TENS4D;
    dx->t4 = tens4D_alloc(cl->x_rows, cl->x_cols,
                          cl->x_depth, cl->batch_size);


    padding_t dy_padding = { cl->filter_size - 1 - cl->padding[TOP],
                             cl->filter_size - 1 - cl->padding[BOTTOM],
                             cl->filter_size - 1 - cl->padding[LEFT],
                             cl->filter_size - 1 - cl->padding[RIGHT] };

    tens4D dy_padded = tens4D_alloc(cl->y_rows + dy_padding[TOP] + dy_padding[BOTTOM],
                                    cl->y_cols + dy_padding[LEFT] + dy_padding[RIGHT],
                                    cl->convolutions, cl->batch_size);
    tens4D_pad(dy_padded, dy.t4, dy_padding);

    tens4D filter_180 = tens4D_alloc(cl->filter_size, cl->filter_size,
                                     cl->x_depth, cl->convolutions);

    tens4D_180(filter_180, cl->filters);

    tens4D_fill(dx->t4, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->x_depth; ++j) {
            mat convolved = mat_alloc(cl->x_rows, cl->x_cols);

            for (int k = 0; k < cl->convolutions; ++k) {
                mat_convolve(convolved, dy_padded.tens3Ds[i].mats[k],
                             filter_180.tens3Ds[k].mats[j]);
                mat_add(dx->t4.tens3Ds[i].mats[j],
                        dx->t4.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    tens4D x_padded = tens4D_alloc(cl->x_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                   cl->x_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                   cl->x_depth, cl->batch_size);
    tens4D_pad(x_padded, cl->x_cache, cl->padding);

    tens4D dw = tens4D_alloc(cl->filter_size, cl->filter_size,
                             cl->x_depth, cl->convolutions);
    tens4D_fill(dw, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->x_depth; ++j) {
            mat convolved = mat_alloc(cl->filter_size, cl->filter_size);

            for (int k = 0; k < cl->batch_size; ++k) {
                mat_convolve(convolved, x_padded.tens3Ds[k].mats[j],
                             dy.t4.tens3Ds[k].mats[i]);
                mat_add(dw.tens3Ds[i].mats[j], dw.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    tens4D_scale(dw, dw, rate / cl->batch_size);
    tens4D_func(dw, dw, clip);
    tens4D_sub(cl->filters, cl->filters, dw);

    tens3D db = tens3D_alloc(cl->y_rows, cl->y_cols, cl->convolutions);
    tens3D_fill(db, 0);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->y_rows; ++j) {
            for (int k = 0; k < cl->y_cols; ++k) {
                float sum = 0;

                for (int l = 0; l < cl->batch_size; ++l) {
                    sum += tens4D_at(dy.t4, j, k, i, l);
                }

                tens3D_at(db, j, k, i) = sum;
            }
        }
    }

    tens3D_scale(db, db, rate / cl->batch_size);
    tens3D_func(db, db, clip);
    tens3D_sub(cl->b, cl->b, db);

    tens4D_destroy(dy_padded);
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

    free(cl);
}

void conv_init(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_normal(cl->filters, 0, sqrt(2.0 / (cl->x_depth * cl->filter_size * cl->filter_size)));

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
