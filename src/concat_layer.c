#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer concat_layer_alloc(int input_size, int batch_size, int steps)
{
    concat_layer *cl = malloc(sizeof(concat_layer));
    cl->input_size = input_size;
    cl->output_size = input_size * steps;
    cl->batch_size = batch_size;
    cl->steps = steps;

    layer l;
    l.type = CONCAT;
    l.data = cl;
    l.forward = concat_forward;
    l.backprop = concat_backprop;
    l.destroy = concat_destroy;
    l.he = NULL;
    l.glorot = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void concat_forward(layer l, void *input, void **output)
{
    concat_layer *cl = (concat_layer *)l.data;

    tens3D *tens3D_input = (tens3D *)input;

    assert(tens3D_input->rows == cl->input_size);
    assert(tens3D_input->cols == cl->batch_size);
    assert(tens3D_input->depth == cl->steps);

    mat *mat_output = malloc(sizeof(mat));
    *mat_output = mat_alloc(cl->output_size, cl->batch_size);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < cl->steps; ++i) {
        for (int j = 0; j < cl->input_size; ++j) {
            for (int k = 0; k < cl->batch_size; ++k) {
                mat_at(*mat_output, i * cl->input_size + j, k) = tens3D_at(*tens3D_input, j, k, i);
            }
        }
    }

    *output = mat_output;
}

void concat_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    concat_layer *cl = (concat_layer *)l.data;

    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == cl->output_size);
    assert(mat_grad_in->cols == cl->batch_size);

    tens3D *tens3D_grad_out = malloc(sizeof(tens3D));
    *tens3D_grad_out = tens3D_alloc(cl->input_size, cl->batch_size, cl->steps);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < cl->steps; ++i) {
        for (int j = 0; j < cl->input_size; ++j) {
            for (int k = 0; k < cl->batch_size; ++k) {
                tens3D_at(*tens3D_grad_out, j, k, i) = mat_at(*mat_grad_in, i * cl->input_size + j, k);
            }
        }
    }

    *grad_out = tens3D_grad_out;
}

void concat_destroy(layer l)
{
    concat_layer *cl = (concat_layer *)l.data;

    free(cl);
}
