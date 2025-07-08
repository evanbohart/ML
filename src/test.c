#include "nn.h"
#include "utils.h"

int main(void)
{
    nn net = nn_alloc(3);
    nn_add_layer(&net, recurrent_layer_alloc(1, 1, 1, 10, 5, TANH, TANH));
    nn_add_layer(&net, concat_layer_alloc(1, 10, 5));
    nn_add_layer(&net, dense_layer_alloc(5, 1, 10, RELU));
    nn_glorot(net);

    mat targets = mat_alloc(1, 10);
    for (int i = 0; i < 10; ++i) {
        mat_at(targets, 0, i) = i;
    }

    tens3D input = tens3D_alloc(1, 10, 5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 10; ++j) {
            tens3D_at(input, 0, j, i) = j;
        }
    }

    mat grad_in = mat_alloc(1, 10);

    void *output;
    void *grad_out;

    for (int i = 0; i < 100 * 1000; ++i) {
    	output = NULL;
	    grad_out = NULL;

        nn_forward(net, &input, &output);
        mat *predicted = (mat *)output;

        mat_sub(grad_in, *predicted, targets);
        mat_print(grad_in);
        printf("\n");

        nn_backprop(net, &grad_in, &grad_out, 1e-4);
        tens3D *grad_input = (tens3D *)grad_out;

        free(predicted->vals);
        tens3D_destroy(*grad_input);

        free(output);
        free(grad_out);
    }

    free(targets.vals);
    tens3D_destroy(input);
    free(grad_in.vals);

    return 0;
}
