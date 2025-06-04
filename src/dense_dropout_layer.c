#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include "utils.h"

layer dense_dropout_layer_alloc(int input_size, int batch_size, double rate)
{
    dense_dropout_layer *ddl = malloc(sizeof(dense_dropout_layer));

    ddl->input_size = input_size;
    ddl->batch_size = batch_size;
    ddl->rate = rate;
    ddl->mask = mat_alloc(input_size, batch_size);

    layer l;
    l.type = DENSE_DROPOUT;
    l.data = ddl;
    l.forward = dense_dropout_forward;
    l.backprop = dense_dropout_backprop;
    l.destroy = dense_dropout_destroy;
    l.glorot = NULL;
    l.he = NULL;
    l.print = NULL;

    return l;
}

void dense_dropout_forward(layer l, void *inputs, void **outputs)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;
    mat *mat_inputs = (mat *)inputs;

    assert(mat_inputs->rows == ddl->input_size);
    assert(mat_inputs->cols == ddl->batch_size);

    mat *mat_outputs = malloc(sizeof(mat));

    for (int i = 0; i < ddl->input_size; ++i) {
        for (int j = 0; j < ddl->batch_size; ++j) {
            mat_at(ddl->mask, i, j) = rand_double(0, 1) > ddl->rate ? 1 : 0;
        }
    }

    *mat_outputs = mat_alloc(ddl->input_size, ddl->batch_size);
    mat_had(*mat_outputs, *mat_inputs, ddl->mask);

    *outputs = mat_outputs;
}

void dense_dropout_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;
    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == ddl->input_size);
    assert(mat_grad_in->cols == ddl->batch_size);

    mat *mat_grad_out = malloc(sizeof(mat));
    *mat_grad_out = mat_alloc(ddl->input_size, ddl->batch_size);

    mat_had(*mat_grad_out, *mat_grad_in, ddl->mask);

    *grad_out = mat_grad_out;
}

void dense_dropout_destroy(layer l)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;

    free(ddl->mask.vals);
    free(ddl);
}
