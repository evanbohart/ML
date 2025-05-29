#include <stdlib.h>
#include <string.h>
#include <assert.h>
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

    cl->biases = tens_alloc(conv_rows, conv_cols, input_channels);

    cl->activation = activation;
    cl->input_rows = input_rows;
    cl->input_cols = input_cols;
    cl->input_channels = input_channels;
    cl->conv_rows = conv_rows;
    cl->conv_cols = conv_cols;
    cl->output_rows = conv_rows / pooling_size;
    cl->output_cols = conv_rows / pooling_size;
    cl->output_channels = convolutions;
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

    return l;
}

void conv_forward(layer l, void *inputs, void **outputs)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens *tens_in = (tens *)inputs;
    tens *tens_out = malloc(sizeof(tens));

    tens padded = tens_alloc(cl->input_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                             cl->input_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                             cl->input_channels);
    tens_pad(padded, *tens_in, cl->padding);

    mat convolved = mat_alloc(cl->conv_rows, cl->conv_cols);

    tens_fill(cl->lins_cache, 0);

    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat_convolve(convolved, padded.mats[j], cl->filters[i].mats[j]);
            mat_add(cl->lins_cache.mats[i], cl->lins_cache.mats[i], convolved);
        }
    }

    tens_add(cl->lins_cache, cl->lins_cache, cl->biases);

    tens activated = tens_alloc(cl->output_rows, cl->output_cols, cl->convolutions);

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
 
    *tens_out = tens_alloc(cl->output_rows / cl->pooling_size,
                           cl->output_cols / cl->pooling_size,
                           cl->convolutions);

    tens_maxpool(*tens_out, cl->pooling_mask, activated, cl->pooling_size);

    *outputs = tens_out;
}

void conv_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens *tens_in = (tens *)grad_in;
    tens *tens_out = malloc(sizeof(tens));

    tens lins_deriv = tens_alloc(cl->lins_cache.rows, cl->lins_cache.cols, cl->convolutions);
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

    padding_t padding = { cl->filter_size - 1, cl->filter_size - 1, 
                          cl->filter_size - 1, cl->filter_size - 1 };

    tens padded = tens_alloc(grad.rows + padding[TOP] + padding[BOTTOM],
                             grad.cols + padding[LEFT] + padding[RIGHT],
                             cl->convolutions);
    tens_pad(padded, grad, padding);

    mat filter_trans = mat_alloc(cl->filter_size, cl->filter_size);
    mat convolved = mat_alloc(cl->input_rows, cl->input_cols);

    *tens_out = tens_alloc(cl->input_rows, cl->input_cols, cl->convolutions);
    tens_fill(*tens_out, 0);

    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat_trans(filter_trans, cl->filters[i].mats[j]);
            mat_convolve(convolved, padded.mats[j], filter_trans);

            mat_add(tens_out->mats[i], tens_out->mats[i], convolved);
        }
    }

    *grad_out = tens_out;

    for (int i = 0; i < cl->convolutions; ++i) {
        tens dw = tens_alloc(cl->filters[i].rows, cl->filters[i].cols, cl->filters[i].depth);

        for (int j = 0; j < cl->input_channels; ++j) {
            mat_convolve(dw.mats[j], cl->input_cache.mats[j], grad.mats[i]);
        }

        tens_scale(dw, dw, rate);
        tens_sub(cl->filters[i], cl->filters[i], dw);

        tens_destroy(&dw);
    }

    tens db = tens_alloc(grad.rows, grad.cols, cl->convolutions);
    tens_copy(db, grad);

    tens_scale(db, db, rate);
    tens_sub(cl->biases, cl->biases, db);

    tens_destroy(&lins_deriv);
    tens_destroy(&unpooled);
    tens_destroy(&grad);
    tens_destroy(&padded);
    free(filter_trans.vals);
    free(convolved.vals);
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

