#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

#define SUB_LAYERS 2

block res_block_alloc(int x_rows, int x_cols, int x_depth,
                      int batch_size, int convolutions,
                      int filter_size, int stride)
{
    res_block *rb = malloc(sizeof(res_block));

    rb->x_rows = x_rows;
    rb->x_cols = x_cols;
    rb->x_depth = x_depth;
    rb->x_batches = batch_size;
    rb->convolutions = convolutions;

    rb->skip = tens4D_alloc(x_rows, x_cols,
                            convolutions, batch_size);

    int padding_width = (filter_size - 1);
    int padding_height = (filter_size - 1);

    padding_t proj_padding = { 0, 0, 0, 0 };

    padding_t padding;
    padding[TOP] = padding_height / 2;
    padding[BOTTOM] = padding_height - padding[TOP];
    padding[LEFT] = padding_width / 2;
    padding[RIGHT] = padding_width - padding[LEFT];

    rb->conv_layers = malloc(SUB_LAYERS * sizeof(layer));
    rb->batchnorm_layers = malloc(SUB_LAYERS * sizeof(layer));
    rb->relu_layers = malloc(SUB_LAYERS * sizeof(layer));

    rb->proj_layer = conv_layer_alloc(x_rows, x_cols, x_depth,
                                      batch_size, convolutions,
                                      1, 1, proj_padding);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[0] = conv_layer_alloc(x_rows, x_cols, convolutions,
                                            batch_size, convolutions,
                                            filter_size, stride, padding);
        rb->batchnorm_layers[0] = batchnorm_layer_4D_alloc(x_rows, x_cols,
                                                        convolutions, batch_size);
        rb->relu_layers[0] = relu_layer_4D_alloc(x_rows, x_cols,
                                                convolutions, batch_size);
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

    assert(x.rows == rb->x_rows);
    assert(x.cols == rb->x_cols);
    assert(x.depth == rb->x_depth);
    assert(x.batches == rb->x_batches);

    tens x_current = x;
    tens y_current;

    rb->proj_layer.forward(rb->proj_layer, x_current, &y_current);

    tens_copy(rb->skip, y_current);

    x_current = y_current;

    rb->conv_layers[0].forward(rb->conv_layers[0], x_current, &y_current);

    tens_destroy(x_current);
    x_current = y_current;

    rb->batchnorm_layers[0].forward(rb->batchnorm_layers[0], x_current, &y_current);

    tens_destroy(x_current);
    x_current = y_current;

    rb->relu_layers[0].forward(rb->relu_layers[0], x_current, &y_current);

    tens_destroy(x_current);
    x_current = y_current;

    rb->conv_layers[1].forward(rb->conv_layers[1], x_current, &y_current);

    tens_destroy(x_current);
    x_current = y_current;

    rb->batchnorm_layers[1].forward(rb->batchnorm_layers[1], x_current, &y_current);

    tens_destroy(x_current);
    x_current = y_current;

    tens_add(x_current, x_current, rb->skip);

    rb->relu_layers[1].forward(rb->relu_layers[1], x_current, &y_current);

    tens_destroy(x_current);

    *y = y_current;
}

void res_backprop(block b, tens dy, tens *dx, float rate)
{
    res_block *rb = (res_block *)b.data;

    assert(dy.rows == rb->x_rows);
    assert(dy.cols == rb->x_cols);
    assert(dy.depth == rb->convolutions);
    assert(dy.batches == rb->x_batches);

    tens dy_current = dy;
    tens dx_current;

    tens dskip = tens4D_alloc(rb->x_rows, rb->x_cols, rb->convolutions, rb->x_batches);

    rb->relu_layers[1].backprop(rb->relu_layers[1], dy_current, &dx_current, rate);

    tens_copy(dskip, dx_current);

    dy_current = dx_current;

    rb->batchnorm_layers[1].backprop(rb->batchnorm_layers[1], dy_current, &dx_current, rate);

    tens_destroy(dy_current);
    dy_current = dx_current;

    rb->conv_layers[1].backprop(rb->conv_layers[1], dy_current, &dx_current, rate);

    tens_destroy(dy_current);
    dy_current = dx_current;

    rb->relu_layers[0].backprop(rb->relu_layers[0], dy_current, &dx_current, rate);

    tens_destroy(dy_current);
    dy_current = dx_current;

    rb->batchnorm_layers[0].backprop(rb->batchnorm_layers[0], dy_current, &dx_current, rate);

    tens_destroy(dy_current);
    dy_current = dx_current;

    rb->conv_layers[0].backprop(rb->conv_layers[0], dy_current, &dx_current, rate);

    tens_destroy(dy_current);
    dy_current = dx_current;

    tens_add(dy_current, dy_current, dskip);

    rb->proj_layer.backprop(rb->proj_layer, dy_current, &dx_current, rate);

    tens_destroy(dy_current);

    *dx = dx_current;

    tens_destroy(dskip);
}

void res_destroy(block b)
{
    res_block *rb = (res_block *)b.data;

    rb->proj_layer.destroy(rb->proj_layer);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[i].destroy(rb->conv_layers[i]);
        rb->batchnorm_layers[i].destroy(rb->batchnorm_layers[i]);
        rb->relu_layers[i].destroy(rb->relu_layers[i]);
    }

    free(rb);
}

void res_init(block b)
{
    res_block *rb = (res_block *)b.data;

    rb->proj_layer.init(rb->proj_layer);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[0].init(rb->conv_layers[i]);
        rb->batchnorm_layers[0].init(rb->batchnorm_layers[i]);
    }
}

void res_print(block b)
{
    res_block *rb = (res_block *)b.data;

    rb->proj_layer.print(rb->proj_layer);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[0].print(rb->conv_layers[0]);
        rb->batchnorm_layers[0].print(rb->batchnorm_layers[0]);
    }
}

void res_save(block b, FILE *f)
{
    res_block *rb = (res_block *)b.data;

    rb->proj_layer.save(rb->proj_layer, f);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[i].save(rb->conv_layers[i], f);
        rb->batchnorm_layers[i].save(rb->batchnorm_layers[i], f);
    }
}

void res_load(block b, FILE *f)
{
    res_block *rb = (res_block *)b.data;

    rb->proj_layer.load(rb->proj_layer, f);

    for (int i = 0; i < SUB_LAYERS; ++i) {
        rb->conv_layers[i].load(rb->conv_layers[i], f);
        rb->batchnorm_layers[i].load(rb->batchnorm_layers[i], f);
    }
}
