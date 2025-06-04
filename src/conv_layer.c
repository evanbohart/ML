#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

layer conv_layer_alloc(int input_rows, int input_cols, int input_channels,
                       int batch_size, int filter_size, int convolutions, int stride,
                       padding_t padding, int pooling_size, actfunc activation)
{
    int conv_rows = (input_rows + padding[TOP] + padding[BOTTOM] - filter_size) / stride + 1;
    int conv_cols = (input_cols + padding[LEFT] + padding[RIGHT] - filter_size) / stride + 1;

    assert(conv_rows % pooling_size == 0);
    assert(conv_cols % pooling_size == 0);

    conv_layer *cl = malloc(sizeof(conv_layer));

    cl->filters = tens4D_alloc(filter_size, filter_size, input_channels, convolutions);
    cl->biases = tens3D_alloc(conv_rows, conv_cols, convolutions);

    cl->activation = activation;
    cl->input_rows = input_rows;
    cl->input_cols = input_cols;
    cl->input_channels = input_channels;
    cl->batch_size = batch_size;
    cl->conv_rows = conv_rows;
    cl->conv_cols = conv_cols;
    cl->output_rows = conv_rows / pooling_size;
    cl->output_cols = conv_cols / pooling_size;
    cl->filter_size = filter_size;
    cl->convolutions = convolutions;
    cl->stride = stride;
    memcpy(cl->padding, padding, sizeof(cl->padding));
    cl->pooling_size = pooling_size;

    cl->input_cache = tens4D_alloc(input_rows, input_cols, input_channels, batch_size);
    cl->lins_cache = tens4D_alloc(conv_rows, conv_cols, convolutions, batch_size);
    cl->pooling_mask = tens4D_alloc(conv_rows, conv_cols, convolutions, batch_size);

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
    tens4D *tens4D_inputs = (tens4D *)inputs;
    tens4D *tens4D_outputs = malloc(sizeof(tens4D));

    tens4D_copy(cl->input_cache, *tens4D_inputs);

    tens4D inputs_padded = tens4D_alloc(cl->input_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                        cl->input_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                        cl->input_channels, cl->batch_size);
    tens4D_pad(inputs_padded, *tens4D_inputs, cl->padding);

    mat convolved = mat_alloc(cl->conv_rows, cl->conv_cols);

    tens4D_fill(cl->lins_cache, 0);

    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            for (int k = 0; k < cl->convolutions; ++k) {
                mat_convolve(convolved, inputs_padded.tens3Ds[i].mats[j],
                             cl->filters.tens3Ds[k].mats[j]);
                mat_add(cl->lins_cache.tens3Ds[i].mats[k],
                        cl->lins_cache.tens3Ds[i].mats[k], convolved);
            }
        }
    }

    for (int i = 0; i < cl->conv_rows; ++i) {
        for (int j = 0; j < cl->conv_cols; ++j) {
            for (int k = 0; k < cl->convolutions; ++k) {
                for (int l = 0; l < cl->batch_size; ++l) {
                    tens4D_at(cl->lins_cache, i, j, k, l) += tens3D_at(cl->biases, i, j, k);
                }
            }
        }
    }

    tens4D activated = tens4D_alloc(cl->conv_rows, cl->conv_cols,
                                    cl->convolutions, cl->batch_size);

    switch (cl->activation) {
        case SIGMOID:
            tens4D_func(activated, cl->lins_cache, sig);
            break;
        case RELU:
            tens4D_func(activated, cl->lins_cache, relu);
            break;
        case SOFTMAX:
            break;
    }

    *tens4D_outputs = tens4D_alloc(cl->output_rows, cl->output_cols,
                                   cl->convolutions, cl->batch_size);
    tens4D_maxpool(*tens4D_outputs, activated, cl->pooling_mask, cl->pooling_size);

    *outputs = tens4D_outputs;

    tens4D_destroy(inputs_padded);
    tens4D_destroy(activated);
}

void conv_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;
    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    tens4D lins_deriv = tens4D_alloc(cl->conv_rows, cl->conv_cols,
                                     cl->convolutions, cl->batch_size);

    switch (cl->activation) {
        case SIGMOID:
            tens4D_func(lins_deriv, cl->lins_cache, dsig);
            break;
        case RELU:
            tens4D_func(lins_deriv, cl->lins_cache, drelu);
            break;
        case SOFTMAX:
            break;
    }

    tens4D unpooled = tens4D_alloc(cl->conv_rows, cl->conv_cols,
                                   cl->convolutions, cl->batch_size);
    tens4D_maxunpool(unpooled, *tens4D_grad_in, cl->pooling_mask, cl->pooling_size);

    tens4D grad = tens4D_alloc(cl->conv_rows, cl->conv_cols,
                               cl->convolutions, cl->batch_size);
    tens4D_had(grad, unpooled, lins_deriv);

    padding_t padding = { cl->filter_size - 1 - cl->padding[TOP],
                          cl->filter_size - 1 - cl->padding[BOTTOM],
                          cl->filter_size - 1 - cl->padding[LEFT],
                          cl->filter_size - 1 - cl->padding[RIGHT] };

    tens4D grad_padded = tens4D_alloc(cl->conv_rows + padding[TOP] + padding[BOTTOM],
                                      cl->conv_cols + padding[LEFT] + padding[RIGHT],
                                      cl->convolutions, cl->batch_size);
    tens4D_pad(grad_padded, grad, padding);

    tens4D filter_trans = tens4D_alloc(cl->filter_size, cl->filter_size,
                                       cl->input_channels, cl->convolutions);
    tens4D filter_180 = tens4D_alloc(cl->filter_size, cl->filter_size,
                                     cl->input_channels, cl->convolutions);

    tens4D_trans(filter_trans, cl->filters);
    tens4D_trans(filter_180, filter_trans);

    *tens4D_grad_out = tens4D_alloc(cl->input_rows, cl->input_cols,
                                    cl->input_channels, cl->batch_size);
    tens4D_fill(*tens4D_grad_out, 0);

    mat grad_out_convolved = mat_alloc(cl->input_rows, cl->input_cols);

    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            for (int k = 0; k < cl->convolutions; ++k) {
                mat_convolve(grad_out_convolved, grad_padded.tens3Ds[i].mats[k],
                             filter_180.tens3Ds[k].mats[j]);
                mat_add(tens4D_grad_out->tens3Ds[i].mats[j],
                        tens4D_grad_out->tens3Ds[i].mats[j], grad_out_convolved);
            }
        }
    }

    *grad_out = tens4D_grad_out;

    tens4D inputs_padded = tens4D_alloc(cl->input_rows + padding[TOP] + padding[BOTTOM],
                                        cl->input_cols + padding[LEFT] + padding[RIGHT],
                                        cl->input_channels, cl->batch_size);
    tens4D_pad(inputs_padded, cl->input_cache, padding);

    tens4D dw = tens4D_alloc(cl->filter_size, cl->filter_size,
                             cl->input_channels, cl->convolutions);
    tens4D_fill(dw, 0);

    mat dw_convolved = mat_alloc(cl->filter_size, cl->filter_size);

    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            for (int k = 0; k < cl->convolutions; ++k) {
                mat_convolve(dw_convolved, inputs_padded.tens3Ds[i].mats[j],
                             grad.tens3Ds[i].mats[k]);
                mat_add(dw.tens3Ds[k].mats[j], dw.tens3Ds[k].mats[j], dw_convolved);
            }
        }
    }

    tens4D_scale(dw, dw, 1 / cl->batch_size);
    tens4D_func(dw, dw, clip);
    tens4D_scale(dw, dw, rate);
    tens4D_sub(cl->filters, cl->filters, dw);

    tens3D db = tens3D_alloc(cl->conv_rows, cl->conv_cols, cl->convolutions);
    tens3D_fill(db, 0);

    for (int i = 0; i < cl->conv_rows; ++i) {
        for (int j = 0; j < cl->conv_cols; ++j) {
            for (int k = 0; k < cl->convolutions; ++k) {
                for (int l = 0; l < cl->batch_size; ++l) {
                    tens3D_at(db, i, j, k) += tens4D_at(grad, i, j, k, l);
                }
            }
        }
    }

    tens3D_scale(db, db, 1 / cl->batch_size);
    tens3D_func(db, db, clip);
    tens3D_scale(db, db, rate);
    tens3D_sub(cl->biases, cl->biases, db);

    tens4D_destroy(lins_deriv);
    tens4D_destroy(unpooled);
    tens4D_destroy(grad);
    tens4D_destroy(grad_padded);
    tens4D_destroy(filter_trans);
    tens4D_destroy(filter_180);
    free(grad_out_convolved.vals);
    tens4D_destroy(inputs_padded);
    free(dw_convolved.vals);
    tens4D_destroy(dw);
    tens3D_destroy(db);
}

void conv_destroy(layer l)
{
   conv_layer *cl = (conv_layer *)l.data;

    tens4D_destroy(cl->filters);
    tens3D_destroy(cl->biases);
    tens4D_destroy(cl->input_cache);
    tens4D_destroy(cl->lins_cache);
    tens4D_destroy(cl->pooling_mask);

    free(cl);
}

void conv_he(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_normal(cl->filters, 0, sqrt(2.0 / (cl->input_channels * cl->filter_size * cl->filter_size)));
    tens3D_fill(cl->biases, 0);
}

void conv_glorot(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_normal(cl->filters, 0, sqrt(2.0 / (cl->input_channels * cl->filter_size * cl->filter_size +
                                                 cl->convolutions * cl->filter_size * cl->filter_size)));
    tens3D_fill(cl->biases, 0);
}

void conv_print(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_print(cl->filters);
    tens3D_print(cl->biases);
}
