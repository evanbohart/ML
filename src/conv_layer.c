#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer conv_layer_alloc(int input_rows, int input_cols, int input_channels,
                       int batch_size, int filter_size, int convolutions,
                       int stride, padding_t padding, actfunc activation)
{
    conv_layer *cl = malloc(sizeof(conv_layer));

    int output_rows = (input_rows + padding[TOP] + padding[BOTTOM] - filter_size) / stride + 1;
    int output_cols = (input_cols + padding[LEFT] + padding[RIGHT] - filter_size) / stride + 1;

    cl->filters = tens4D_alloc(filter_size, filter_size, input_channels, convolutions);
    cl->biases = tens3D_alloc(output_rows, output_cols, convolutions);

    cl->activation = activation;
    cl->input_rows = input_rows;
    cl->input_cols = input_cols;
    cl->input_channels = input_channels;
    cl->batch_size = batch_size;
    cl->output_rows = output_rows;
    cl->output_cols = output_cols;
    cl->filter_size = filter_size;
    cl->convolutions = convolutions;
    cl->stride = stride;
    memcpy(cl->padding, padding, sizeof(cl->padding));

    cl->input_cache = tens4D_alloc(input_rows, input_cols, input_channels, batch_size);
    cl->lins_cache = tens4D_alloc(output_rows, output_cols, convolutions, batch_size);

    layer l;
    l.type = CONV;
    l.data = cl;
    l.forward = conv_forward;
    l.backprop = conv_backprop;
    l.destroy = conv_destroy;
    l.he = conv_he;
    l.glorot = conv_glorot;
    l.print = conv_print;
    l.save = conv_save;
    l.load = conv_load;

    return l;
}

void conv_forward(layer l, void *inputs, void **outputs)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens4D *tens4D_inputs = (tens4D *)inputs;

    assert(tens4D_inputs->rows == cl->input_rows);
    assert(tens4D_inputs->cols == cl->input_cols);
    assert(tens4D_inputs->depth == cl->input_channels);
    assert(tens4D_inputs->batches == cl->batch_size);

    tens4D *tens4D_outputs = malloc(sizeof(tens4D));

    tens4D_copy(cl->input_cache, *tens4D_inputs);

    tens4D inputs_padded = tens4D_alloc(cl->input_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                        cl->input_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                        cl->input_channels, cl->batch_size);
    tens4D_pad(inputs_padded, *tens4D_inputs, cl->padding);

    tens4D_fill(cl->lins_cache, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            mat convolved = mat_alloc(cl->output_rows, cl->output_cols);

            for (int k = 0; k < cl->input_channels; ++k) {
                mat_convolve(convolved, inputs_padded.tens3Ds[i].mats[k],
                             cl->filters.tens3Ds[j].mats[k]);
                mat_add(cl->lins_cache.tens3Ds[i].mats[j],
                        cl->lins_cache.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            for (int k = 0; k < cl->output_rows; ++k) {
                for (int l = 0; l < cl->output_cols; ++l) {
                    tens4D_at(cl->lins_cache, k, l, j, i) += tens3D_at(cl->biases, k, l, j);
                }
            }
        }
    }

    *tens4D_outputs = tens4D_alloc(cl->output_rows, cl->output_cols,
                                    cl->convolutions, cl->batch_size);

    switch (cl->activation) {
        case SIGMOID:
            tens4D_func(*tens4D_outputs, cl->lins_cache, sig);
            break;
        case RELU:
            tens4D_func(*tens4D_outputs, cl->lins_cache, relu);
            break;
        case SOFTMAX:
            break;
    }

    *outputs = tens4D_outputs;

    tens4D_destroy(inputs_padded);
}

void conv_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    conv_layer *cl = (conv_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;

    assert(tens4D_grad_in->rows == cl->output_rows);
    assert(tens4D_grad_in->cols == cl->output_cols);
    assert(tens4D_grad_in->depth == cl->convolutions);
    assert(tens4D_grad_in->batches == cl->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    tens4D lins_deriv = tens4D_alloc(cl->output_rows, cl->output_cols,
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

    tens4D grad = tens4D_alloc(cl->output_rows, cl->output_cols,
                               cl->convolutions, cl->batch_size);
    tens4D_had(grad, *tens4D_grad_in, lins_deriv);

    padding_t grad_padding = { cl->filter_size - 1 - cl->padding[TOP],
                               cl->filter_size - 1 - cl->padding[BOTTOM],
                               cl->filter_size - 1 - cl->padding[LEFT],
                               cl->filter_size - 1 - cl->padding[RIGHT] };

    tens4D grad_padded = tens4D_alloc(cl->output_rows + grad_padding[TOP] + grad_padding[BOTTOM],
                                      cl->output_cols + grad_padding[LEFT] + grad_padding[RIGHT],
                                      cl->convolutions, cl->batch_size);
    tens4D_pad(grad_padded, grad, grad_padding);

    tens4D filter_180 = tens4D_alloc(cl->filter_size, cl->filter_size,
                                     cl->input_channels, cl->convolutions);

    tens4D_180(filter_180, cl->filters);

    *tens4D_grad_out = tens4D_alloc(cl->input_rows, cl->input_cols,
                                    cl->input_channels, cl->batch_size);
    tens4D_fill(*tens4D_grad_out, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->batch_size; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat convolved = mat_alloc(cl->input_rows, cl->input_cols);

            for (int k = 0; k < cl->convolutions; ++k) {
                mat_convolve(convolved, grad_padded.tens3Ds[i].mats[k],
                             filter_180.tens3Ds[k].mats[j]);
                mat_add(tens4D_grad_out->tens3Ds[i].mats[j],
                        tens4D_grad_out->tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    *grad_out = tens4D_grad_out;

    tens4D inputs_padded = tens4D_alloc(cl->input_rows + cl->padding[TOP] + cl->padding[BOTTOM],
                                        cl->input_cols + cl->padding[LEFT] + cl->padding[RIGHT],
                                        cl->input_channels, cl->batch_size);
    tens4D_pad(inputs_padded, cl->input_cache, cl->padding);

    tens4D dw = tens4D_alloc(cl->filter_size, cl->filter_size,
                             cl->input_channels, cl->convolutions);
    tens4D_fill(dw, 0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->input_channels; ++j) {
            mat convolved = mat_alloc(cl->filter_size, cl->filter_size);

            for (int k = 0; k < cl->batch_size; ++k) {
                mat_convolve(convolved, inputs_padded.tens3Ds[k].mats[j],
                             grad.tens3Ds[k].mats[i]);
                mat_add(dw.tens3Ds[i].mats[j], dw.tens3Ds[i].mats[j], convolved);
            }

            free(convolved.vals);
        }
    }

    tens4D_scale(dw, dw, 1.0 / cl->batch_size);
    tens4D_func(dw, dw, clip);
    tens4D_scale(dw, dw, rate);
    tens4D_sub(cl->filters, cl->filters, dw);

    tens3D db = tens3D_alloc(cl->output_rows, cl->output_cols, cl->convolutions);
    tens3D_fill(db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->output_rows; ++j) {
            for (int k = 0; k < cl->output_cols; ++k) {
                float sum = 0;

                for (int l = 0; l < cl->batch_size; ++l) {
                    sum += tens4D_at(grad, j, k, i, l);
                }

                tens3D_at(db, j, k, i) = sum;
            }
        }
    }

    tens3D_scale(db, db, 1.0 / cl->batch_size);
    tens3D_func(db, db, clip);
    tens3D_scale(db, db, rate);
    tens3D_sub(cl->biases, cl->biases, db);

    tens4D_destroy(lins_deriv);
    tens4D_destroy(grad);
    tens4D_destroy(grad_padded);
    tens4D_destroy(filter_180);
    tens4D_destroy(inputs_padded);
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

void conv_save(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_save(cl->filters, f);
    tens3D_save(cl->biases, f);
}

void conv_load(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens4D_load(cl->filters, f);
    tens3D_load(cl->biases, f);
}
