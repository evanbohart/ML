#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer flatten_layer_alloc(int input_rows, int input_cols,
                          int input_channels, int batch_size)
{
    flatten_layer *fl = malloc(sizeof(flatten_layer));
    fl->input_rows = input_rows;
    fl->input_cols = input_cols;
    fl->input_channels = input_channels;
    fl->batch_size = batch_size;
    fl->output_size = input_rows * input_cols * input_channels;

    layer l;
    l.type = FLATTEN;
    l.data = fl;
    l.forward = flatten_forward;
    l.backprop = flatten_backprop;
    l.destroy = flatten_destroy;
    l.glorot = NULL;
    l.he = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void flatten_forward(layer l, void *inputs, void **outputs)
{
    flatten_layer *fl = (flatten_layer *)l.data;
    tens4D *tens4D_inputs = (tens4D *)inputs;

    assert(tens4D_inputs->rows == fl->input_rows);
    assert(tens4D_inputs->cols == fl->input_cols);
    assert(tens4D_inputs->depth == fl->input_channels);
    assert(tens4D_inputs->batches == fl->batch_size);

    mat *mat_outputs = malloc(sizeof(mat));
    *mat_outputs = mat_alloc(fl->output_size, fl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->input_channels; ++j) {
            for (int k = 0; k < fl->input_rows; ++k) {
                for (int l = 0; l < fl->input_cols; ++l) {
                    int index = (j * fl->input_rows + k) * fl->input_cols + l;
                    mat_at(*mat_outputs, index, i) = tens4D_at(*tens4D_inputs, k, l, j, i);
                }
            }
        }
    }

    *outputs = mat_outputs;
}

void flatten_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    flatten_layer *fl = (flatten_layer *)l.data;
    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == fl->output_size);
    assert(mat_grad_in->cols == fl->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(fl->input_rows, fl->input_cols,
                                    fl->input_channels, fl->batch_size);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->input_channels; ++j) {
            for (int k = 0; k < fl->input_rows; ++k) {
                for (int l = 0; l < fl->input_cols; ++l) {
                    int index = (j * fl->input_rows + k) * fl->input_cols + l;
                    tens4D_at(*tens4D_grad_out, k, l, j, i) = mat_at(*mat_grad_in, index, i);
                }
            }
        }
    }

    *grad_out = tens4D_grad_out;
}

void flatten_destroy(layer l)
{
    flatten_layer *fl = (flatten_layer *)l.data;

    free(fl);
}
