#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

layer conv_layer_alloc(int input_rows, int input_cols, int input_channels,
                       int filter_size, int convolutions, int stride,
                       padding_t padding, int pooling_size, actfunc activation)
{
    int conv_rows = (input_rows + padding[TOP] + padding[BOTTOM] - filter_size) / stride + 1;
    int conv_cols = (input_cols + padding[LEFT] + padding[RIGHT] - filter_size) / stride + 1;

    assert(conv_rows % pooling_size == 0);
    assert(conv_cols % pooling_size == 0);

    conv_layer *cl = malloc(sizeof(conv_layer));

    cl->filters = malloc(convolutions * sizeof(tens));

    for (int i = 0; i < convolutions; ++i) {
        cl->filters[i] = tens_alloc(filter_size, filter_size, input_channels);
    }

    cl->biases = tens_alloc(conv_rows, conv_cols, convolutions);

    cl->activation = activation;
    cl->input_rows = input_rows;
    cl->input_cols = input_cols;
    cl->input_channels = input_channels;
    cl->conv_rows = conv_rows;
    cl->conv_cols = conv_cols;
    cl->output_rows = conv_rows / pooling_size;
    cl->output_cols = conv_cols / pooling_size;
    cl->filter_size = filter_size;
    cl->convolutions = convolutions;
    cl->stride = stride;
    memcpy(cl->padding, padding, sizeof(cl->padding));
    cl->pooling_size = pooling_size;
    cl->input_cache = tens_alloc(input_rows, input_cols, input_channels);

    cl->lins_cache = tens_alloc(conv_rows, conv_cols, convolutions);
    cl->pooling_mask = tens_alloc(conv_rows, conv_cols, convolutions);

    layer l;
    l.type = CONV;
    l.data = cl;
    l.forward = conv_forward;
    l.backprop = conv_backprop;
    l.destroy = conv_destroy;
    l.he = conv_he;
    l.glorot = conv_glorot;
    l.print = conv_print;

    return l;
}

void conv_forward(layer l, void *inputs, void **outputs)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens *tens_in = (tens *)inputs;
    tens *tens_out = malloc(sizeof(tens));

    tens grad_padded = tens_alloc(cl->input_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                             cl->input_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                             cl->input_channels);
    tens_pad(grad_padded, *tens_in, cl->padding);

    mat convolved = mat_alloc(cl->conv_rows, cl->conv_cols);

    tens_fill(cl->lins_cache, 0);

    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat_convolve(convolved, grad_padded.mats[j], cl->filters[i].mats[j]);
            mat_add(cl->lins_cache.mats[i], cl->lins_cache.mats[i], convolved);
        }
    }

    tens_add(cl->lins_cache, cl->lins_cache, cl->biases);

    tens activated = tens_alloc(cl->conv_rows, cl->conv_cols, cl->convolutions);

    switch (cl->activation) {
        case SIGMOID:
            tens_func(activated, cl->lins_cache, sig);
            break;
        case RELU:
            tens_func(activated, cl->lins_cache, relu);
            break;
        case SOFTMAX:
            break;
    }
 
    *tens_out = tens_alloc(cl->output_rows, cl->output_cols, cl->convolutions);
    tens_maxpool(*tens_out, cl->pooling_mask, activated, cl->pooling_size);

    *outputs = tens_out;
}

void conv_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens *tens_in = (tens *)grad_in;
    tens *tens_out = malloc(sizeof(tens));

    tens lins_deriv = tens_alloc(cl->conv_rows, cl->conv_cols, cl->convolutions);

    switch (cl->activation) {
        case SIGMOID:
            tens_func(lins_deriv, cl->lins_cache, dsig);
            break;
        case RELU:
            tens_func(lins_deriv, cl->lins_cache, drelu);
            break;
        case SOFTMAX:
            break;
    }

    tens unpooled = tens_alloc(cl->conv_rows, cl->conv_cols, cl->convolutions);
    tens_maxunpool(unpooled, cl->pooling_mask, *tens_in, cl->pooling_size);

    tens grad = tens_alloc(cl->conv_rows, cl->conv_cols, cl->convolutions);
    tens_had(grad, unpooled, lins_deriv);

    padding_t padding = { cl->filter_size - 1 - cl->padding[TOP],
                          cl->filter_size - 1 - cl->padding[BOTTOM],
                          cl->filter_size - 1 - cl->padding[LEFT],
                          cl->filter_size - 1 - cl->padding[RIGHT] };

    tens grad_padded = tens_alloc(cl->conv_rows + padding[TOP] + padding[BOTTOM],
                             cl->conv_cols + padding[LEFT] + padding[RIGHT],
                             cl->convolutions);
    tens_pad(grad_padded, grad, padding);

    mat filter_trans = mat_alloc(cl->filter_size, cl->filter_size);
    mat filter_180 = mat_alloc(cl->filter_size, cl->filter_size);
    mat convolved = mat_alloc(cl->input_rows, cl->input_cols);

    *tens_out = tens_alloc(cl->input_rows, cl->input_cols, cl->input_channels);
    tens_fill(*tens_out, 0);

    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat_trans(filter_trans, cl->filters[i].mats[j]);
            mat_trans(filter_180, filter_trans);
            mat_convolve(convolved, grad_padded.mats[i], filter_180);
            mat_add(tens_out->mats[j], tens_out->mats[j], convolved);
        }
    }

    *grad_out = tens_out;

    tens inputs_padded = tens_alloc(cl->input_rows + padding[TOP] + padding[BOTTOM],
                                    cl->input_cols + padding[LEFT] + padding[RIGHT],
                                    cl->input_channels);
    tens_pad(inputs_padded, cl->input_cache, padding);

    for (int i = 0; i < cl->convolutions; ++i) {
        tens dw = tens_alloc(cl->filters[i].rows, cl->filters[i].cols, cl->filters[i].depth);

        for (int j = 0; j < cl->input_channels; ++j) {
            mat_convolve(dw.mats[j], inputs_padded.mats[j], grad.mats[i]);
        }

        tens_func(dw, dw, clip);
        tens_scale(dw, dw, rate);
        tens_sub(cl->filters[i], cl->filters[i], dw);

        tens_destroy(&dw);
    }

    tens db = tens_alloc(grad.rows, grad.cols, cl->convolutions);
    tens_copy(db, grad);

    tens_func(db, db, clip);
    tens_scale(db, db, rate);
    tens_sub(cl->biases, cl->biases, db);

    tens_destroy(&lins_deriv);
    tens_destroy(&unpooled);
    tens_destroy(&grad);
    tens_destroy(&grad_padded);
    free(filter_trans.vals);
    free(filter_180.vals);
    free(convolved.vals);
    tens_destroy(&inputs_padded);
    tens_destroy(&db);
}

void conv_destroy(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    for (int i = 0; i < cl->convolutions; ++i) {
        tens_destroy(&cl->filters[i]);
    }

    free(cl->filters);

    tens_destroy(&cl->biases);
    tens_destroy(&cl->input_cache);
    tens_destroy(&cl->lins_cache);

    free(cl);
}

void conv_he(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    for (int i = 0; i < cl->convolutions; ++i) {
        tens_normal(cl->filters[i], 0, sqrt(2.0 / (cl->input_channels * cl->filter_size * cl->filter_size)));
    }

    tens_fill(cl->biases, 0);
}

void conv_glorot(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    for (int i = 0; i < cl->convolutions; ++i) {
        tens_normal(cl->filters[i], 0, sqrt(2.0 / (cl->input_channels * cl->filter_size * cl->filter_size +
                                                   cl->convolutions * cl->filter_size * cl->filter_size)));
    }

    tens_fill(cl->biases, 0);
}

void conv_print(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    for (int i = 0; i < cl->convolutions; ++i) {
        tens_print(cl->filters[i]);
    }

    tens_print(cl->biases);
}
