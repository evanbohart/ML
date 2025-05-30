#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

double sig(double x) { return 1 / (1 + exp(x)); }

double dsig(double x) { return sig(x) * (1 - sig(x)); }

double relu(double x) { return x * (x > 0); }

double drelu(double x) { return x > 0; }

double clip(double x) {
    if (x > 1) return 1;
    if (x < -1) return -1;
    return x;
}

nn nn_alloc(int max_layers)
{
    nn n;

    n.num_layers = 0;
    n.max_layers = max_layers;
    n.layers = malloc(max_layers * sizeof(layer));

    return n;
}

void nn_add_layer(nn *n, layer l)
{
    assert(n->num_layers != n->max_layers);

    if (n->num_layers > 0) {
        layer *prev = &n->layers[n->num_layers - 1];

        layer_type prev_type = prev->type;
        layer_type new_type = l.type;

        assert(prev_type != DENSE || new_type != CONV);

        if (prev_type == CONV && new_type == CONV) {
            conv_layer *cl_prev = (conv_layer *)prev->data;
            conv_layer *cl_new = (conv_layer *)l.data;

            assert(cl_prev->output_rows == cl_new->input_rows);
            assert(cl_prev->output_cols == cl_new->input_cols);
            assert(cl_prev->convolutions == cl_new->input_channels);
        }
        else if (prev_type == CONV && new_type == DENSE) {
            conv_layer *cl_prev = (conv_layer *)prev->data;
            dense_layer *dl_new = (dense_layer *)l.data;

            assert(cl_prev->output_rows * cl_prev->output_cols *
                   cl_prev->convolutions == dl_new->input_size);
        }
        else {
            dense_layer *dl_prev = (dense_layer *)prev->data;
            dense_layer *dl_new = (dense_layer *)l.data;

            assert(dl_prev->output_size == dl_new->input_size);
        }
    }

    n->layers[n->num_layers++] = l;
}

void nn_forward(nn n, void *inputs, void **outputs)
{
    void *current_inputs = inputs;
    void *current_outputs;

    for (int i = 0; i < n.num_layers; ++i) {
        current_outputs = NULL;

        if (i > 0 && n.layers[i - 1].type == CONV && n.layers[i].type == DENSE) {
            tens *inputs_tens = (tens *)current_inputs;
            mat *flattened = malloc(sizeof(mat));
            *flattened = mat_alloc(inputs_tens->rows * inputs_tens->cols * inputs_tens->depth, 1);

            tens_flatten(*flattened, *inputs_tens);

            current_inputs = flattened;
        }

        n.layers[i].forward(n.layers[i], current_inputs, &current_outputs);

        if (i > 0) free(current_inputs);

        current_inputs = current_outputs;
    }

    *outputs = current_outputs;
}

void nn_backprop(nn n, void *grad_in, void **grad_out, double rate)
{
    void *current_grad_in = grad_in;
    void *current_grad_out;

    for (int i = n.num_layers - 1; i >= 0; --i) {
        current_grad_out = NULL;

        if (i < n.num_layers - 1 && n.layers[i + 1].type == DENSE && n.layers[i].type == CONV) {
            mat *grad_in_mat = (mat *)current_grad_in;
            tens *unflattened = malloc(sizeof(tens));

            conv_layer *cl = (conv_layer *)n.layers[i].data;
            *unflattened = tens_alloc(cl->output_rows, cl->output_cols, cl->convolutions);

            mat_unflatten(*unflattened, *grad_in_mat);

            current_grad_in = unflattened;
        }

        n.layers[i].backprop(n.layers[i], current_grad_in, &current_grad_out, rate); 

        if (i < n.num_layers - 1) free(current_grad_in);

        current_grad_in = current_grad_out;
    }

    *grad_out = current_grad_out;
}

void nn_destroy(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].destroy(n.layers[i]);
    }

    free(n.layers);
}

void nn_he(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].he(n.layers[i]);
    }
}

void nn_glorot(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].glorot(n.layers[i]);
    }
}

void nn_print(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].print(n.layers[i]);
    }
}
