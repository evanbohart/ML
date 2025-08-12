#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "nn.h"

layer maxpool_layer_alloc(int x_rows, int x_cols, int x_depth,
                          int batch_size, int pooling_size)
{
    assert(x_rows % pooling_size == 0);
    assert(x_cols % pooling_size == 0);

    maxpool_layer *ml = malloc(sizeof(maxpool_layer));
    ml->x_rows = x_rows;
    ml->x_cols = x_cols;
    ml->x_depth = x_depth;
    ml->batch_size = batch_size;
    ml->pooling_size = pooling_size;
    ml->y_rows = x_rows / pooling_size;
    ml->y_cols = x_cols / pooling_size;

    ml->mask = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);

    layer l;
    l.type = MAXPOOL;
    l.data = ml;
    l.forward = maxpool_forward;
    l.backprop = maxpool_backprop;
    l.destroy = maxpool_destroy;
    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void maxpool_forward(layer l, void *input, void **output)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;
    tens4D *tens4D_input = (tens4D *)input;

    assert(tens4D_input->rows == ml->x_rows);
    assert(tens4D_input->cols == ml->x_cols);
    assert(tens4D_input->depth == ml->x_depth);
    assert(tens4D_input->batches == ml->batch_size);

    tens4D *tens4D_output = malloc(sizeof(tens4D));
    *tens4D_output = tens4D_alloc(ml->y_rows, ml->y_cols,
                                   ml->x_depth, ml->batch_size);

    tens4D_fill(ml->mask, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->batch_size; ++i) {
        for (int j = 0; j < ml->x_depth; ++j) {
            for (int k = 0; k < ml->y_rows; ++k) {
                for (int l = 0; l < ml->y_cols; ++l) {
                    float max = -FLT_MAX;
                    int max_row = 0;
                    int max_col = 0;

                    for (int m = 0; m < ml->pooling_size; ++m) {
                        for (int n = 0; n < ml->pooling_size; ++n) {
                            int row = k * ml->pooling_size + m;
                            int col = l * ml->pooling_size + n;
                            float val = tens4D_at(*tens4D_input, row, col, j, i);
                            if (val > max) {
                                max = val;
                                max_row = m;
                                max_col = n;
                            }
                        }
                    }

                    tens4D_at(*tens4D_output, k, l, j, i) = max;

                    int mask_row = k * ml->pooling_size + max_row;
                    int mask_col = l * ml->pooling_size + max_col;

                    tens4D_at(ml->mask, mask_row, mask_col, j, i) = 1.0f;
                }
            }
        }
    }

    *output = tens4D_output;
}

void maxpool_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;

    assert(tens4D_grad_in->rows == ml->y_rows);
    assert(tens4D_grad_in->cols == ml->y_cols);
    assert(tens4D_grad_in->depth == ml->x_depth);
    assert(tens4D_grad_in->batches == ml->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(ml->x_rows, ml->x_cols,
                                    ml->x_depth, ml->batch_size);

    tens4D_fill(*tens4D_grad_out, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->batch_size; ++i) {
        for (int j = 0; j < ml->x_depth; ++j) {
            for (int k = 0; k < ml->y_rows; ++k) {
                for (int l = 0; l < ml->y_cols; ++l) {
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
