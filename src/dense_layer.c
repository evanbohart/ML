#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "nn.h"

layer dense_layer_alloc(int input_size, int output_size,
                        int batch_size, actfunc activation)
{
    dense_layer *dl = malloc(sizeof(dense_layer));
    dl->weights = mat_alloc(output_size, input_size);
    dl->biases = mat_alloc(output_size, 1);
    dl->activation = activation;
    dl->input_size = input_size;
    dl->output_size = output_size;
    dl->batch_size = batch_size;
    dl->input_cache = mat_alloc(input_size, batch_size);
    dl->lins_cache = mat_alloc(output_size, batch_size);

    layer l;
    l.type = DENSE;
    l.data = dl;
    l.forward = dense_forward;
    l.backprop = dense_backprop;
    l.destroy = dense_destroy;
    l.he = dense_he;
    l.glorot = dense_glorot;
    l.print = dense_print;
    l.save = dense_save;
    l.load = dense_load;

    return l;
}

void dense_forward(layer l, void *input, void **output)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_input = (mat *)input;

    assert(mat_input->rows == dl->input_size);
    assert(mat_input->cols == dl->batch_size);

    mat *mat_output = malloc(sizeof(mat));
    *mat_output = mat_alloc(dl->output_size, dl->batch_size);

    mat_copy(dl->input_cache, *mat_input);

    mat_dot(dl->lins_cache, dl->weights, dl->input_cache);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->output_size; ++i) {
        for (int j = 0; j < dl->batch_size; ++j) {
            mat_at(dl->lins_cache, i, j) += mat_at(dl->biases, i, 0);
        }
    }

    switch (dl->activation) {
        case LIN:
            mat_func(*mat_output, dl->lins_cache, lin);
            break;
        case SIG:
            mat_func(*mat_output, dl->lins_cache, sig);
            break;
        case TANH:
            mat_func(*mat_output, dl->lins_cache, tanhf);
            break;
        case RELU:
            mat_func(*mat_output, dl->lins_cache, relu);
            break;
    }

    *output = mat_output;
}

void dense_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_grad_in = (mat *)grad_in;

    assert(mat_grad_in->rows == dl->output_size);
    assert(mat_grad_in->cols == dl->batch_size);

    mat *mat_grad_out = malloc(sizeof(mat));
    *mat_grad_out = mat_alloc(dl->input_size, dl->batch_size);

    mat lins_deriv = mat_alloc(dl->output_size, dl->batch_size);
    mat grad = mat_alloc(dl->output_size, dl->batch_size);

    switch (dl->activation) {
        case LIN:
            mat_func(lins_deriv, dl->lins_cache, dlin);
            break;
        case SIG:
            mat_func(lins_deriv, dl->lins_cache, dsig);
            break;
        case TANH:
            mat_func(lins_deriv, dl->lins_cache, dtanh);
            break;
        case RELU:
            mat_func(lins_deriv, dl->lins_cache, drelu);
            break;
    }

    mat_had(grad, *mat_grad_in, lins_deriv);

    mat input_trans = mat_alloc(dl->batch_size, dl->input_size);
    mat_trans(input_trans, dl->input_cache);

    mat dw = mat_alloc(dl->output_size, dl->input_size);
    mat_dot(dw, grad, input_trans);

    mat db = mat_alloc(dl->output_size, 1);
    mat_fill(db, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dl->output_size; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < dl->batch_size; ++j) {
            sum += mat_at(grad, i, j);
        }

        mat_at(db, i, 0) = sum;
    }

    mat_scale(dw, dw, 1.0 / dl->batch_size);
    mat_func(dw, dw, clip);
    mat_scale(dw, dw, rate);
    mat_sub(dl->weights, dl->weights, dw);

    mat_scale(db, db, 1.0 / dl->batch_size);
    mat_func(db, db, clip);
    mat_scale(db, db, rate);
    mat_sub(dl->biases, dl->biases, db);

    mat weights_trans = mat_alloc(dl->weights.cols, dl->weights.rows);
    mat_trans(weights_trans, dl->weights);

    mat_dot(*mat_grad_out, weights_trans, grad);

    *grad_out = mat_grad_out;

    free(lins_deriv.vals);
    free(grad.vals);
    free(input_trans.vals);
    free(dw.vals);
    free(db.vals);
    free(weights_trans.vals);
}

void dense_destroy(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    free(dl->weights.vals);
    free(dl->biases.vals);
    free(dl->input_cache.vals);
    free(dl->lins_cache.vals);

    free(dl);
}

void dense_he(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_normal(dl->weights, 0, sqrt(2.0 / dl->input_size));
    mat_fill(dl->biases, 0);
}

void dense_glorot(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_normal(dl->weights, 0, sqrt(2.0 / (dl->input_size + dl->output_size)));
    mat_fill(dl->biases, 0);
}

void dense_print(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_print(dl->weights);
    mat_print(dl->biases);
}

void dense_save(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_save(dl->weights, f);
    mat_save(dl->biases, f);
}

void dense_load(layer l, FILE *f)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_load(dl->weights, f);
    mat_load(dl->biases, f);
}
