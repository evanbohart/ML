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
            assert(cl_prev->batch_size == cl_new->batch_size);
        }
        else if (prev_type == CONV && new_type == DENSE) {
            conv_layer *cl_prev = (conv_layer *)prev->data;
            dense_layer *dl_new = (dense_layer *)l.data;

            assert(cl_prev->output_rows * cl_prev->output_cols *
                   cl_prev->convolutions == dl_new->input_size);
            assert(cl_prev->batch_size == dl_new->batch_size);
        }
        else {
            dense_layer *dl_prev = (dense_layer *)prev->data;
            dense_layer *dl_new = (dense_layer *)l.data;

            assert(dl_prev->output_size == dl_new->input_size);
            assert(dl_prev->batch_size == dl_new->batch_size);
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
            tens4D *inputs_tens4D = (tens4D *)current_inputs;
            mat *flattened = malloc(sizeof(mat));
            *flattened = mat_alloc(inputs_tens4D->rows * inputs_tens4D->cols * inputs_tens4D->depth,
                                   inputs_tens4D->batches);

            tens4D_flatten(*flattened, *inputs_tens4D);

            tens4D_destroy(*inputs_tens4D);
            free(current_inputs);

            current_inputs = flattened;
        }

        n.layers[i].forward(n.layers[i], current_inputs, &current_outputs);

        if (i > 0) {
            switch (n.layers[i].type) {
                case DENSE:
                    mat *inputs_mat = (mat *)current_inputs;
                    free(inputs_mat->vals);
                    break;
                    case CONV:
                    tens4D *inputs_tens4D = (tens4D *)current_inputs;
                    tens4D_destroy(*inputs_tens4D);
                    break;
            }

            free(current_inputs);
        }

        current_inputs = current_outputs;
    }

    *outputs = current_inputs;
}

void nn_backprop(nn n, void *grad_in, void **grad_out, double rate)
{
    void *current_grad_in = grad_in;
    void *current_grad_out;

    for (int i = n.num_layers - 1; i >= 0; --i) {
        current_grad_out = NULL;

        if (i < n.num_layers - 1 && n.layers[i + 1].type == DENSE && n.layers[i].type == CONV) {
            mat *grad_in_mat = (mat *)current_grad_in;
            tens4D *unflattened = malloc(sizeof(tens4D));

            conv_layer *cl = (conv_layer *)n.layers[i].data;
            *unflattened = tens4D_alloc(cl->output_rows, cl->output_cols,
                                        cl->convolutions, cl->batch_size);

            mat_unflatten(*unflattened, *grad_in_mat);

            free(grad_in_mat->vals);
            free(current_grad_in);

            current_grad_in = unflattened;
        }

        n.layers[i].backprop(n.layers[i], current_grad_in, &current_grad_out, rate);

        if (i < n.num_layers - 1) {
            switch (n.layers[i].type) {
                case DENSE:
                    mat *mat_grad_in = (mat *)current_grad_in;
                    free(mat_grad_in->vals);
                    break;
                case CONV:
                    tens4D *tens4D_grad_in = (tens4D *)current_grad_in;
                    tens4D_destroy(*tens4D_grad_in);
                    break;
            }

            free(current_grad_in);
        }

        current_grad_in = current_grad_out;
    }

    *grad_out = current_grad_in;
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
