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
    void *current_input = inputs;
    void *current_output;

    for (int i = 0; i < n.num_layers; ++i) {
        current_output = NULL;

        switch (n.layers[i].type) {
            case DENSE:
                if (i > 0 && n.layers[i - 1].type == CONV) {
                    tens *tens_input = (tens *)current_input;
                    mat flattened = mat_alloc(tens_input->rows * tens_input->cols * tens_input->depth, 1);
                    tens_flatten(flattened, *tens_input);
                    current_input = &flattened;
                }

                dense_forward(n.layers[i], current_input, &current_output);
                break;
            case CONV:
                conv_forward(n.layers[i], current_input, &current_output);
                break;
        }

        if (i > 0) free(current_input);

        current_input = current_output;
    }

    *outputs = current_output;
}

void nn_backprop(nn n, void *grad_in, void **grad_out, double rate);
void nn_destroy(nn n);
