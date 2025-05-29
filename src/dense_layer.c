#include <stdlib.h>
#include "nn.h"

layer dense_layer_alloc(int input_size, int output_size, actfunc activation)
{
    dense_layer *dl = malloc(sizeof(dense_layer));

    dl->weights = mat_alloc(output_size, input_size);
    dl->biases = mat_alloc(output_size, 1);
    dl->activation = activation;
    dl->input_size = input_size;
    dl->output_size = output_size;
    dl->input_cache = mat_alloc(input_size, 1);
    dl->lins_cache = mat_alloc(output_size, 1);

    layer l;
    l.type = DENSE;
    l.data = dl;
    l.forward = dense_forward;
    l.backprop = dense_backprop;
    l.destroy = dense_destroy;

    return l;
}

void dense_forward(layer l, void *inputs, void **outputs)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_in = (mat *)inputs;
    mat *mat_out = malloc(sizeof(mat));

    mat_copy(dl->input_cache, *mat_in);

	mat_dot(dl->lins_cache, dl->weights, dl->input_cache);
	mat_add(dl->lins_cache, dl->lins_cache, dl->biases);

    *mat_out = mat_alloc(dl->output_size, dl->lins_cache.cols);

    switch (dl->activation) {
        case SIGMOID:
            mat_func(*mat_out, dl->lins_cache, sig);
            break;
        case RELU:
            mat_func(*mat_out, dl->lins_cache, relu);
            break;
        case SOFTMAX:
            mat_softmax(*mat_out, dl->lins_cache);
            break;
    }

    *outputs = mat_out;
}

void dense_backprop(layer l, void *grad_in, void **grad_out, double rate)
{
    dense_layer *dl = (dense_layer *)l.data;
    mat *mat_in = (mat *)grad_in;
    mat *mat_out = malloc(sizeof(mat));

    mat lins_deriv = mat_alloc(dl->output_size, 1);

    switch (dl->activation) {
        case SIGMOID:
            mat_func(lins_deriv, dl->lins_cache, dsig);
            break;
        case RELU:
            mat_func(lins_deriv, dl->lins_cache, drelu);
            break;
        case SOFTMAX:
            break;
    }

    mat grad_current = mat_alloc(lins_deriv.rows, lins_deriv.cols);
    mat_had(grad_current, *mat_in, lins_deriv);

    mat input_trans = mat_alloc(1, dl->input_size);
    mat_trans(input_trans, dl->input_cache);

    mat dw = mat_alloc(grad_current.rows, input_trans.cols);
    mat_dot(dw, grad_current, input_trans);
    mat_scale(dw, dw, rate);

    mat db = mat_alloc(grad_current.rows, grad_current.cols);
    mat_copy(db, grad_current);
    mat_scale(db, db, rate);

    mat_sub(dl->weights, dl->weights, dw);
    mat_sub(dl->biases, dl->biases, db);

    mat weights_trans = mat_alloc(dl->weights.cols, dl->weights.rows);
    mat_trans(weights_trans, dl->weights);

    *mat_out = mat_alloc(weights_trans.rows, grad_current.cols);
    mat_dot(*mat_out, weights_trans, grad_current);

    *grad_out = mat_out;

    free(lins_deriv.vals);
    free(grad_current.vals);
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

    free(dl);
}

void dense_layer_glorot(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_normal(dl->weights, 0, 2 / (dl->input_size + dl->output_size));
    mat_fill(dl->biases, 0);
}

void dense_layer_he(layer l)
{
    dense_layer *dl = (dense_layer *)l.data;

    mat_normal(dl->weights, 0, 2 / dl->input_size);
    mat_fill(dl->biases, 0);
}
