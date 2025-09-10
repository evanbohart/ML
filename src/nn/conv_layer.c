#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int batch_size, int convolutions,
                       int filter_size, int stride, padding_t x_padding)
{
    conv_layer *cl = malloc(sizeof(conv_layer));

    int y_rows = (x_rows + x_padding[TOP] + x_padding[BOTTOM] -
                  filter_size) / stride + 1;
    int y_cols = (x_cols + x_padding[LEFT] + x_padding[RIGHT] -
                  filter_size) / stride + 1;

    cl->x_rows = x_rows;
    cl->x_cols = x_cols;
    cl->x_depth = x_depth;
    cl->x_batches = batch_size;

    cl->y_rows = y_rows;
    cl->y_cols = y_cols;
    cl->convolutions = convolutions;

    cl->filter_size = filter_size;
    cl->stride = stride;

    memcpy(cl->x_padding, x_padding, sizeof(cl->x_padding));

    cl->dy_padding[TOP] = filter_size - 1 - x_padding[TOP];
    cl->dy_padding[BOTTOM] = filter_size - 1 - x_padding[BOTTOM];
    cl->dy_padding[LEFT] = filter_size - 1 - x_padding[LEFT];
    cl->dy_padding[RIGHT] = filter_size - 1 - x_padding[RIGHT];

    cl->w = tens4D_alloc(filter_size, filter_size, x_depth, convolutions);
    cl->b = tens3D_alloc(y_rows, y_cols, convolutions);

    cl->x_padded = tens4D_alloc(x_rows + x_padding[TOP] + x_padding[BOTTOM],
                                x_cols + x_padding[LEFT] + x_padding[RIGHT],
                                x_depth, batch_size);
    cl->dy_padded = tens4D_alloc(y_rows + cl->dy_padding[TOP] + cl->dy_padding[BOTTOM],
                                 y_cols + cl->dy_padding[LEFT] + cl->dy_padding[RIGHT],
                                 convolutions, batch_size);
    cl->w_180 = tens4D_alloc(filter_size, filter_size, x_depth, convolutions);

    cl->dw = tens4D_alloc(filter_size, filter_size, x_depth, convolutions);
    cl->db = tens3D_alloc(y_rows, y_cols, convolutions);

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

    assert(x.rows == cl->x_rows);
    assert(x.cols == cl->x_cols);
    assert(x.depth == cl->x_depth);
    assert(x.batches == cl->x_batches);

    *y = tens4D_alloc(cl->y_rows, cl->y_cols, cl->convolutions, cl->x_batches);

    tens_pad(cl->x_padded, x, cl->x_padding);

    tens_convolve(*y, x, cl->w);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->x_batches; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            for (int k = 0; k < cl->y_rows; ++k) {
                for (int l = 0; l < cl->y_cols; ++l) {
                    tens4D_at(*y, k, l, j, i) += tens3D_at(cl->b, k, l, j);
                }
            }
        }
    }
}

void conv_backprop(layer l, tens dy, tens *dx, float rate)
{
    conv_layer *cl = (conv_layer *)l.data;

    assert(dy.rows == cl->y_rows);
    assert(dy.cols == cl->y_cols);
    assert(dy.depth == cl->convolutions);
    assert(dy.batches == cl->x_batches);

    *dx = tens4D_alloc(cl->x_rows, cl->x_cols, cl->x_depth, cl->x_batches);

    tens_pad(cl->dy_padded, dy, cl->dy_padding);
    tens_180(cl->w_180, cl->w);

    tens_convolve(*dx, cl->dy_padded, cl->w_180);

    tens_convolve(cl->dw, cl->x_padded, dy);

    tens_fill(cl->db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->y_rows; ++j) {
            for (int k = 0; k < cl->y_cols; ++k) {
                float sum = 0;

                for (int l = 0; l < cl->x_batches; ++l) {
                    sum += tens4D_at(dy, j, k, i, l);
                }

                tens3D_at(cl->db, j, k, i) = sum;
            }
        }
    }

    tens_scale(cl->dw, cl->dw, rate / cl->x_batches);
    tens_func(cl->dw, cl->dw, clip);
    tens_sub(cl->w, cl->w, cl->dw);

    tens_scale(cl->db, cl->db, rate / cl->x_batches);
    tens_func(cl->db, cl->db, clip);
    tens_sub(cl->b, cl->b, cl->db);
}

void conv_destroy(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_destroy(cl->w);
    tens_destroy(cl->b);

    tens_destroy(cl->x_padded);
    tens_destroy(cl->dy_padded);
    tens_destroy(cl->w_180);

    tens_destroy(cl->dw);
    tens_destroy(cl->db);

    free(cl);
}

void conv_init(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_normal(cl->w, 0, sqrt(2.0 / (cl->x_depth * cl->filter_size * cl->filter_size)));

    tens_fill(cl->b, 0);
}

void conv_print(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_print(cl->w);
    tens_print(cl->b);
}

void conv_save(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_save(cl->w, f);
    tens_save(cl->b, f);
}

void conv_load(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_load(cl->w, f);
    tens_load(cl->b, f);
}
