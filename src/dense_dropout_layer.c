#include <stdlib.h>
#include <assert.h>
#include "nn.h"
#include "utils.h"

layer dense_dropout_layer_alloc(int x_size, int batch_size, float rate)
{
    dense_dropout_layer *ddl = malloc(sizeof(dense_dropout_layer));

    ddl->x_size = x_size;
    ddl->batch_size = batch_size;
    ddl->rate = rate;
    ddl->mask = mat_alloc(x_size, batch_size);

    layer l;
    l.type = DENSE_DROPOUT;
    l.data = ddl;
    l.forward = dense_dropout_forward;
    l.backprop = dense_dropout_backprop;
    l.destroy = dense_dropout_destroy;
    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void dense_dropout_forward(layer l, void *input, void **output)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;
    mat *mat_input = (mat *)input;

    assert(mat_input->rows == ddl->x_size);
    assert(mat_input->cols == ddl->batch_size);

    mat *mat_output = malloc(sizeof(mat));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ddl->x_size; ++i) {
        for (int j = 0; j < ddl->batch_size; ++j) {
            mat_at(ddl->mask, i, j) = rand_float(0.0f, 1.0f) > ddl->rate ? 1.0f : 0.0f;
        }
    }

    *mat_output = mat_alloc(ddl->x_size, ddl->batch_size);
    mat_had(*mat_output, *mat_input, ddl->mask);

    *output = mat_output;
}

void dense_dropout_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;
    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == ddl->x_size);
    assert(mat_grad_in->cols == ddl->batch_size);

    mat *mat_grad_out = malloc(sizeof(mat));
    *mat_grad_out = mat_alloc(ddl->x_size, ddl->batch_size);

    mat_had(*mat_grad_out, *mat_grad_in, ddl->mask);

    *grad_out = mat_grad_out;
}

void dense_dropout_destroy(layer l)
{
    dense_dropout_layer *ddl = (dense_dropout_layer *)l.data;

    free(ddl->mask.vals);
    free(ddl);
}
