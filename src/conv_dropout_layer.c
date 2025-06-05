#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include "utils.h"

layer conv_dropout_layer_alloc(int input_rows, int input_cols,
                               int input_channels, int batch_size, double rate)
{
    conv_dropout_layer *cdl = malloc(sizeof(conv_dropout_layer));
    cdl->input_rows = input_rows;
    cdl->input_cols = input_cols;
    cdl->input_channels = input_channels;
    cdl->batch_size = batch_size;
    cdl->rate = rate;
    cdl->mask = tens4D_alloc(input_rows, input_cols, input_channels, batch_size);

    layer l;
    l.type = CONV_DROPOUT;
    l.data = cdl;
    l.forward = conv_dropout_forward;
    l.backprop = conv_dropout_backprop;
    l.destroy = conv_dropout_destroy;
    l.glorot = NULL;
    l.he = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void conv_dropout_forward(layer l, void *inputs, void **outputs)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;
    tens4D *tens4D_inputs = (tens4D *)inputs;

    assert(tens4D_inputs->rows == cdl->input_rows);
    assert(tens4D_inputs->cols == cdl->input_cols);
    assert(tens4D_inputs->depth == cdl->input_channels);
    assert(tens4D_inputs->batches == cdl->batch_size);

    tens4D *tens4D_outputs = malloc(sizeof(tens4D));
    *tens4D_outputs = tens4D_alloc(cdl->input_rows, cdl->input_cols,
                                   cdl->input_channels, cdl->batch_size);

    for (int i = 0; i < cdl->input_rows; ++i) {
        for (int j = 0; j < cdl->input_cols; ++j) {
            for (int k = 0; k < cdl->input_channels; ++k) {
                for (int l = 0; l < cdl->batch_size; ++l) {
                    tens4D_at(cdl->mask, i, j, k, l) = rand_double(0, 1) > cdl->rate ? 1 : 0;
                }
            }
        }
    }

    tens4D_had(*tens4D_outputs, *tens4D_inputs, cdl->mask);

    *outputs = tens4D_outputs;
}

void conv_dropout_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;

    assert(tens4D_grad_in->rows == cdl->input_rows);
    assert(tens4D_grad_in->cols == cdl->input_cols);
    assert(tens4D_grad_in->depth == cdl->input_channels);
    assert(tens4D_grad_in->batches == cdl->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(cdl->input_rows, cdl->input_cols,
                                    cdl->input_channels, cdl->batch_size);

    tens4D_had(*tens4D_grad_out, *tens4D_grad_in, cdl->mask);

    *grad_out = tens4D_grad_out;
}

void conv_dropout_destroy(layer l)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;

    tens4D_destroy(cdl->mask);
    free(cdl);
}
