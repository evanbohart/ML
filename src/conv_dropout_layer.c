#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include "utils.h"

layer conv_dropout_layer_alloc(int x_rows, int x_cols, int x_depth,
                               int batch_size, float rate)
{
    conv_dropout_layer *cdl = malloc(sizeof(conv_dropout_layer));
    cdl->x_rows = x_rows;
    cdl->x_cols = x_cols;
    cdl->x_depth = x_depth;
    cdl->batch_size = batch_size;
    cdl->rate = rate;
    cdl->mask = tens4D_alloc(x_rows, x_cols, x_depth, batch_size);

    layer l;
    l.type = CONV_DROPOUT;
    l.data = cdl;
    l.forward = conv_dropout_forward;
    l.backprop = conv_dropout_backprop;
    l.destroy = conv_dropout_destroy;
    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void conv_dropout_forward(layer l, void *input, void **output)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;
    tens4D *tens4D_input = (tens4D *)input;

    assert(tens4D_input->rows == cdl->x_rows);
    assert(tens4D_input->cols == cdl->x_cols);
    assert(tens4D_input->depth == cdl->x_depth);
    assert(tens4D_input->batches == cdl->batch_size);

    tens4D *tens4D_output = malloc(sizeof(tens4D));
    *tens4D_output = tens4D_alloc(cdl->x_rows, cdl->x_cols,
                                   cdl->x_depth, cdl->batch_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cdl->batch_size; ++i) {
        for (int j = 0; j < cdl->x_depth; ++j) {
            for (int k = 0; k < cdl->x_rows; ++k) {
                for (int l = 0; l < cdl->x_cols; ++l) {
                    tens4D_at(cdl->mask, k, l, j, i) = rand_float(0.0f, 1.0f) > cdl->rate ? 1.0f : 0.0f;
                }
            }
        }
    }

    tens4D_had(*tens4D_output, *tens4D_input, cdl->mask);

    *output = tens4D_output;
}

void conv_dropout_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;
    tens4D *tens4D_grad_in = (tens4D *)grad_in;

    assert(tens4D_grad_in->rows == cdl->x_rows);
    assert(tens4D_grad_in->cols == cdl->x_cols);
    assert(tens4D_grad_in->depth == cdl->x_depth);
    assert(tens4D_grad_in->batches == cdl->batch_size);

    tens4D *tens4D_grad_out = malloc(sizeof(tens4D));
    *tens4D_grad_out = tens4D_alloc(cdl->x_rows, cdl->x_cols,
                                    cdl->x_depth, cdl->batch_size);

    tens4D_had(*tens4D_grad_out, *tens4D_grad_in, cdl->mask);

    *grad_out = tens4D_grad_out;
}

void conv_dropout_destroy(layer l)
{
    conv_dropout_layer *cdl = (conv_dropout_layer *)l.data;

    tens4D_destroy(cdl->mask);
    free(cdl);
}
