#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

block res_block_alloc(int sub_layers, int x_rows, int x_cols,
                      int x_depth, int batch_size, int convolutions,
                      int filter_size, int stride)
{
    res_block *rb = malloc(sizeof(res_block));

    rb->sub_layers = sub_layers;
    rb->x_rows = x_rows;
    rb->x_cols = x_cols;
    rb->x_depth = x_depth;
    rb->batch_size = batch_size;

    rb->x_cache = tens4D_alloc(x_rows, x_cols,
                               x_depth, batch_size);

    int padding_width = fmax(0, (x_cols - 1) * stride + filter_size - x_cols);
    int padding_height = fmax(0, (x_rows - 1) * stride + filter_size - x_rows);

    padding_t padding;
    padding[TOP] = padding_height / 2;
    padding[BOTTOM] = padding_height - padding[TOP];
    padding[LEFT] = padding_width / 2;
    padding[RIGHT] = padding_width - padding[LEFT];

    rb->conv_layers = malloc(sub_layers * sizeof(layer));
    rb->batchnorm_layers = malloc(sub_layers * sizeof(layer));
    rb->relu_layers = malloc(sub_layers * sizeof(layer));

    for (int i = 0; i < sub_layers; ++i) {
        rb->conv_layers[i] = conv_layer_alloc(x_rows, x_cols, x_depth,
                                              batch_size, convolutions,
                                              filter_size, stride, padding);
        rb->batchnorm_layers[i] = batchnorm_layer_4D_alloc(x_rows, x_cols,
                                                           x_depth, batch_size);
        rb->relu_layers[i] = relu_layer_4D_alloc(x_rows, x_cols,
                                                 x_depth, batch_size);
    }

    block b;

    b.data = rb;

    b.forward = res_forward;
    b.backprop = res_backprop;
    b.destroy = res_destroy;

    b.init = res_init;
    b.print = res_print;
    b.save = res_save;
    b.load = res_load;

    return b;
}

void res_forward(block b, tens x, tens *y)
{
    res_block *rb = (res_block *)b.data;

    assert(x.type == TENS4D);
    assert(x.t4.rows == rb->x_rows);
    assert(x.t4.cols == rb->x_cols);
    assert(x.t4.depth == rb->x_depth);
    assert(x.t4.batches == rb->batch_size);

    tens4D_copy(rb->x_cache, x.t4);

    tens x_current = x;
    tens y_current;

    for (int i = 0; i < rb->sub_layers - 1; ++i) {
        rb->conv_layers[i].forward(rb->conv_layers[i],
                                   x_current, &y_current);

        if (i > 0) {
            tens4D_destroy(x_current.t4);
        }

        x_current = y_current;

        rb->batchnorm_layers[i].forward(rb->batchnorm_layers[i],
                                        x_current, &y_current);

        tens4D_destroy(x_current.t4);
        x_current = y_current;

        rb->relu_layers[i].forward(rb->relu_layers[i],
                                   x_current, &y_current);

        tens4D_destroy(x_current.t4);
        x_current = y_current;
    }

    rb->conv_layers[rb->sub_layers - 1].forward(rb->conv_layers[rb->sub_layers - 1],
                                                x_current, &y_current);

    tens4D_destroy(x_current.t4);
    x_current = y_current;

    rb->batchnorm_layers[rb->sub_layers - 1].forward(rb->batchnorm_layers[rb->sub_layers - 1],
                                                     x_current, &y_current);

    tens4D_destroy(x_current.t4);
    x_current = y_current;

    tens4D_add(x_current.t4, x_current.t4, rb->x_cache);

    rb->relu_layers[rb->sub_layers - 1].forward(rb->relu_layers[rb->sub_layers - 1],
                                                x_current, &y_current);

    tens4D_destroy(x_current.t4);

    *y = y_current;
}

void res_backprop(block b, tens dy, tens *dx, float rate)
{
    res_block *rb = (res_block *)b.data;

    assert(dy.type == TENS4D);
    assert(dy.t4.rows == rb->x_rows);
    assert(dy.t4.cols == rb->x_cols);
    assert(dy.t4.depth == rb->x_depth);
    assert(dy.t4.batches == rb->batch_size);

    tens dy_current = dy;
    tens dx_current;

    tens4D dskip = tens4D_alloc(rb->x_rows, rb->x_cols,
                                rb->x_depth, rb->batch_size);

    rb->relu_layers[rb->sub_layers - 1].backprop(rb->relu_layers[rb->sub_layers - 1],
                                                 dy_current, &dx_current, rate);

    tens4D_copy(dskip, dx_current.t4);

    dy_current = dx_current;

    rb->batchnorm_layers[rb->sub_layers - 1].backprop(rb->batchnorm_layers[rb->sub_layers - 1],
                                                      dy_current, &dx_current, rate);

    tens4D_destroy(dy_current.t4);
    dy_current = dx_current;

    rb->conv_layers[rb->sub_layers - 1].backprop(rb->conv_layers[rb->sub_layers - 1],
                                                 dy_current, &dx_current, rate);

    tens4D_destroy(dy_current.t4);
    dy_current = dx_current;

    for (int i = rb->sub_layers - 2; i >= 0; --i) {
        rb->relu_layers[i].backprop(rb->relu_layers[i], dy_current,
                                    &dx_current, rate);

        tens4D_destroy(dy_current.t4);
        dy_current = dx_current;

        rb->batchnorm_layers[i].backprop(rb->batchnorm_layers[i], dy_current,
                                         &dx_current, rate);

        tens4D_destroy(dy_current.t4);
        dy_current = dx_current;

        rb->conv_layers[i].backprop(rb->conv_layers[i], dy_current,
                                    &dx_current, rate);

        tens4D_destroy(dy_current.t4);
        dy_current = dx_current;
    }

    tens4D_add(dx_current.t4, dx_current.t4, dskip);

    *dx = dx_current;

    tens4D_destroy(dskip);
}

void res_destroy(block b)
{
    res_block *rb = (res_block *)b.data;

    for (int i = 0; i < rb->sub_layers; ++i) {
        rb->conv_layers[i].destroy(rb->conv_layers[i]);
        rb->batchnorm_layers[i].destroy(rb->batchnorm_layers[i]);
        rb->relu_layers[i].destroy(rb->relu_layers[i]);
    }

    free(rb);
}

void res_init(block b)
{
    res_block *rb = (res_block *)b.data;

    for (int i = 0; i < rb->sub_layers; ++i) {
        rb->conv_layers[i].init(rb->conv_layers[i]);
        rb->batchnorm_layers[i].init(rb->batchnorm_layers[i]);
    }
}

void res_print(block b)
{
    res_block *rb = (res_block *)b.data;

    for (int i = 0; i < rb->sub_layers; ++i) {
        rb->conv_layers[i].print(rb->conv_layers[i]);
        rb->batchnorm_layers[i].print(rb->batchnorm_layers[i]);
    }
}

void res_save(block b, FILE *f)
{
    res_block *rb = (res_block *)b.data;

    for (int i = 0; i < rb->sub_layers; ++i) {
        rb->conv_layers[i].save(rb->conv_layers[i], f);
        rb->batchnorm_layers[i].save(rb->batchnorm_layers[i], f);
    }
}

void res_load(block b, FILE *f)
{
    res_block *rb = (res_block *)b.data;

    for (int i = 0; i < rb->sub_layers; ++i) {
        rb->conv_layers[i].load(rb->conv_layers[i], f);
        rb->batchnorm_layers[i].load(rb->batchnorm_layers[i], f);
    }
}
