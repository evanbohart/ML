#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "nn.h"

layer maxpool_layer_alloc(int input_rows, int input_cols, int input_channels,
                          int batch_size, int pooling_size)
{
    assert(input_rows % pooling_size == 0);
    assert(input_cols % pooling_size == 0);

    maxpool_layer *ml = malloc(sizeof(maxpool_layer));
    ml->input_rows = input_rows;
    ml->input_cols = input_cols;
    ml->input_channels = input_channels;
    ml->batch_size = batch_size;
    ml->pooling_size = pooling_size;
    ml->output_rows = input_rows / pooling_size;
    ml->output_cols = input_cols / pooling_size;

    ml->mask = tens4D_alloc(input_rows, input_cols, input_channels, batch_size);

    layer l;
    l.type = MAXPOOL;
    l.data = ml;
    l.forward = maxpool_forward;
    l.backprop = maxpool_backprop;
    l.destroy = maxpool_destroy;
    l.he = NULL;
    l.glorot = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void maxpool_forward(layer l, void *inputs, void **outputs)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;
    tens4D *tens4D_inputs = (tens4D *)inputs;

    assert(tens4D_inputs->rows == ml->input_rows);
    assert(tens4D_inputs->cols == ml->input_cols);
    assert(tens4D_inputs->depth == ml->input_channels);
    assert(tens4D_inputs->batches == ml->batch_size);

    tens4D *tens4D_outputs = malloc(sizeof(tens4D));
    *tens4D_outputs = tens4D_alloc(ml->output_rows, ml->output_cols,
                                   ml->input_channels, ml->batch_size);

    tens4D_fill(ml->mask, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->batch_size; ++i) {
        for (int j = 0; j < ml->input_channels; ++j) {
            for (int k = 0; k < ml->output_rows; ++k) {
                for (int l = 0; l < ml->output_cols; ++l) {
                    float max = -FLT_MAX;
                    int max_row = 0;
                    int max_col = 0;

                    for (int m = 0; m < ml->pooling_size; ++m) {
                        for (int n = 0; n < ml->pooling_size; ++n) {
                            int row = k * ml->pooling_size + m;
                            int col = l * ml->pooling_size + n;
                            float val = tens4D_at(*tens4D_inputs, row, col, j, i);
                            if (val > max) {
                                max = val;
                                max_row = m;
                                max_col = n;
                            }
                        }
                    }

                    tens4D_at(*tens4D_outputs, k, l, j, i) = max;

                    int mask_row = k * ml->pooling_size + max_row;
                    int mask_col = l * ml->pooling_size + max_col;

                    tens4D_at(ml->mask, mask_row, mask_col, j, i) = 1.0f;
                }
            }
        }
    }

    *outputs = tens4D_outputs;
}

void maxpool_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;

    assert(tens4D_grad_in->rows == ml->output_rows);
    assert(tens4D_grad_in->cols == ml->output_cols);
    assert(tens4D_grad_in->depth == ml->input_channels);
    assert(tens4D_grad_in->batches == ml->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(ml->input_rows, ml->input_cols,
                                    ml->input_channels, ml->batch_size);

    tens4D_fill(*tens4D_grad_out, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->batch_size; ++i) {
        for (int j = 0; j < ml->input_channels; ++j) {
            for (int k = 0; k < ml->output_rows; ++k) {
                for (int l = 0; l < ml->output_cols; ++l) {
                    for (int m = 0; m < ml->pooling_size; ++m) {
                        for (int n = 0; n < ml->pooling_size; ++n) {
                            int row = k * ml->pooling_size + m;
                            int col = l * ml->pooling_size + n;
                            if (tens4D_at(ml->mask, row, col, j, i) == 1.0f) {
                                float val = tens4D_at(*tens4D_grad_in, k, l, j, i);
                                tens4D_at(*tens4D_grad_out, row, col, j, i) = val;
                            }
                        }
                    }
                }
            }
        }
    }

    *grad_out = tens4D_grad_out;
}

void maxpool_destroy(layer l)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;

    tens4D_destroy(ml->mask);
    free(ml);
}
