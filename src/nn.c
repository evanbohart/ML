#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

float lin(float x) { return x; }

float dlin(float x) { return 1; }

float sig(float x) { return 1 / (1 + exp(x)); }

float dsig(float x) { return sig(x) * (1 - sig(x)); }

float dtanh(float x) { return 1 - pow(tanhf(x), 2); }

float relu(float x) { return x * (x > 0); }

float drelu(float x) { return x > 0; }

float clip(float x) {
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
    n->layers[n->num_layers++] = l;
}

void nn_forward(nn n, void *inputs, void **outputs)
{
    void *current_inputs = inputs;
    void *current_outputs;

    for (int i = 0; i < n.num_layers; ++i) {
        current_outputs = NULL;

        n.layers[i].forward(n.layers[i], current_inputs, &current_outputs);

        if (i > 0) {
            if (n.layers[i].type == DENSE || n.layers[i].type == DENSE_DROPOUT) {
                mat *mat_inputs = (mat *)current_inputs;
                free(mat_inputs->vals);
            }
	        else if (n.layers[i].type == RECURRENT || n.layers[i].type == CONCAT) {
		        tens3D *tens3D_inputs = (tens3D *)current_inputs;
    		    tens3D_destroy(*tens3D_inputs);
	        }
            else {
                tens4D *tens4D_inputs = (tens4D *)current_inputs;
                tens4D_destroy(*tens4D_inputs);
            }

            free(current_inputs);
        }

        current_inputs = current_outputs;
    }

    *outputs = current_inputs;
}

void nn_backprop(nn n, void *grad_in, void **grad_out, float rate)
{
    void *current_grad_in = grad_in;
    void *current_grad_out;

    for (int i = n.num_layers - 1; i >= 0; --i) {
        current_grad_out = NULL;

        n.layers[i].backprop(n.layers[i], current_grad_in, &current_grad_out, rate);

        if (i < n.num_layers - 1) {
            if (n.layers[i].type == DENSE || n.layers[i].type == FLATTEN ||
                n.layers[i].type == DENSE_DROPOUT || n.layers[i].type == CONCAT) {
                mat *mat_grad_in = (mat *)current_grad_in;
                free(mat_grad_in->vals);
            }
	        else if (n.layers[i].type == RECURRENT) {
    		    tens3D *tens3D_grad_in = (tens3D *)current_grad_in;
	    	    tens3D_destroy(*tens3D_grad_in);
	        }
            else {
                tens4D *tens4D_grad_in = (tens4D *)current_grad_in;
                tens4D_destroy(*tens4D_grad_in);
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

void nn_init(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        if (n.layers[i].init) {
            n.layers[i].init(n.layers[i]);
        }
    }
}

void nn_print(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        if (n.layers[i].print) {
            n.layers[i].print(n.layers[i]);
        }
    }
}

void nn_save(nn n, FILE *f)
{
    for (int i = 0; i < n.num_layers; ++i) {
        if (n.layers[i].save) {
            n.layers[i].save(n.layers[i], f);
        }
    }
}

void nn_load(nn n, FILE *f)
{
    for (int i = 0; i < n.num_layers; ++i) {
        if (n.layers[i].load) {
            n.layers[i].load(n.layers[i], f);
        }
    }
}
