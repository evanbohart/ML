#include "nn.h"
#include "cube.h"

#define BATCH_SIZE 32
#define EPOCHS 1

void get_inputs(cube c, tens3D inputs)
{
    const int pos[8][2] = {
        {0, 0},
        {1, 0},
        {2, 0},
        {2, 1},
        {2, 2},
        {1, 2},
        {0, 2},
        {0, 1}
    };

    printf("here");

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            tens3D_at(inputs, pos[j][0], pos[j][1], i) = (c.faces[i].bitboard >> ((7 - j) * 8)) & 0xFF;
        }

        tens3D_at(inputs, 1, 1, i) = c.faces[i].center;
    }
}

int main(void)
{
    nn net = nn_alloc(15);

    padding_t same = { 0, 1, 0, 1 };

    nn_add_layer(&net, conv_layer_alloc(3, 3, 6, BATCH_SIZE, 2, 64, 1, same, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(3, 3, 64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, conv_layer_alloc(3, 3, 64, BATCH_SIZE, 2, 64, 1, same, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(3, 3, 64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, conv_layer_alloc(3, 3, 64, BATCH_SIZE, 2, 64, 1, same, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(3, 3, 64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, conv_layer_alloc(3, 3, 64, BATCH_SIZE, 2, 64, 1, same, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(3, 3, 64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, conv_layer_alloc(3, 3, 64, BATCH_SIZE, 2, 64, 1, same, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(3, 3, 64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, flatten_layer_alloc(3, 3, 64, BATCH_SIZE));
    nn_add_layer(&net, dense_layer_alloc(576, 64, BATCH_SIZE, RELU));
    nn_add_layer(&net, dense_dropout_layer_alloc(64, BATCH_SIZE, 0.25));
    nn_add_layer(&net, dense_layer_alloc(64, 1, BATCH_SIZE, SIGMOID));

    tens4D inputs = tens4D_alloc(3, 3, 6, BATCH_SIZE);
    mat grad_in = mat_alloc(1, BATCH_SIZE);

    void *outputs;
    void *grad_out;

    cube c = init_cube();
    get_inputs(c, inputs.tens3Ds[0]);
    tens3D_print(inputs.tens3Ds[0]);
/*
    for (int i = 0; i < EPOCHS; ++i) {
        for (int j = 0; j < 25 * 1000; ++j) {
            for (int k = 0; k < 20; ++k) {
                for (int l = 0; l < BATCH_SIZE; ++l) {
                    cube c = init_cube();
                    scramble_cube(&c, k);
                    get_inputs(c, inputs.tens3Ds[l]);
                }

                nn_forward(net, &inputs, &outputs);
                mat *predicted = (mat *)outputs;

                mat targets = mat_alloc(1, BATCH_SIZE);
                mat_fill(targets, k);

                mat_sub(grad_in, *predicted, targets);

                nn_backprop(net, &grad_in, &grad_out);
                tens4D *grad_inputs = (tens4D *)grad_out;

                free(predicted->vals);
                tens4D_destroy(*grad_inputs);

                free(outputs);
                free(grad_out);
            }
        }
    }

    char *file = "cube.bin";
    char path[FILENAME_MAX];
    get_path(path, file);

    FILE *f = fopen(path, "wb");
    nn_save(net, f);
*/
    return 0;
}
