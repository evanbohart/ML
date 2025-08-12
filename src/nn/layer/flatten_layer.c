#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer flatten_layer_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size)
{
    flatten_layer *fl = malloc(sizeof(flatten_layer));
    fl->x_rows = x_rows;
    fl->x_cols = x_cols;
    fl->x_depth = x_depth;
    fl->batch_size = batch_size;
    fl->y_size = x_rows * x_cols * x_depth;

    layer l;
    l.type = FLATTEN;
    l.data = fl;
    l.forward = flatten_forward;
    l.backprop = flatten_backprop;
    l.destroy = flatten_destroy;
    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void flatten_forward(layer l, void *input, void **output)
{
    flatten_layer *fl = (flatten_layer *)l.data;
    tens4D *tens4D_input = (tens4D *)input;

    assert(tens4D_input->rows == fl->x_rows);
    assert(tens4D_input->cols == fl->x_cols);
    assert(tens4D_input->depth == fl->x_depth);
    assert(tens4D_input->batches == fl->batch_size);

    mat *mat_output = malloc(sizeof(mat));
    *mat_output = mat_alloc(fl->y_size, fl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->x_depth; ++j) {
            for (int k = 0; k < fl->x_rows; ++k) {
                for (int l = 0; l < fl->x_cols; ++l) {
                    int index = (j * fl->x_rows + k) * fl->x_cols + l;
                    mat_at(*mat_output, index, i) = tens4D_at(*tens4D_input, k, l, j, i);
                }
            }
        }
    }

    *output = mat_output;
}

void flatten_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    flatten_layer *fl = (flatten_layer *)l.data;
    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == fl->y_size);
    assert(mat_grad_in->cols == fl->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(fl->x_rows, fl->x_cols,
                                    fl->x_depth, fl->batch_size);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < fl->batch_size; ++i) {
        for (int j = 0; j < fl->x_depth; ++j) {
            for (int k = 0; k < fl->x_rows; ++k) {
                for (int l = 0; l < fl->x_cols; ++l) {
                    int index = (j * fl->x_rows + k) * fl->x_cols + l;
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
