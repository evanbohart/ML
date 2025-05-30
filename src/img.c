#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "utils.h"

int read_next_img(FILE *f, tens input)
{
    unsigned char target;
    if (!fread(&target, 1, sizeof(target), f)) return -1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                unsigned char val;
                fread(&val, 1, sizeof(val), f);
                tens_at(input, k, j, i) = val;
            }
        }
    }

    return target;
}

int main(void)
{
    srand(time(0));

    char files[5][100];
    get_path(files[0], "cifar-10-batches-bin\\data_batch_1.bin");
    get_path(files[1], "cifar-10-batches-bin\\data_batch_2.bin");
    get_path(files[2], "cifar-10-batches-bin\\data_batch_3.bin");
    get_path(files[3], "cifar-10-batches-bin\\data_batch_4.bin");
    get_path(files[4], "cifar-10-batches-bin\\data_batch_5.bin");

    nn net = nn_alloc(8);
    padding_t same = { 1, 1, 1, 1 };

    nn_add_layer(&net, conv_layer_alloc(32, 32, 3, 3, 32, 1, same, 1, RELU));
    nn_add_layer(&net, conv_layer_alloc(32, 32, 32, 3, 32, 1, same, 2, RELU));
    nn_add_layer(&net, conv_layer_alloc(16, 16, 32, 3, 64, 1, same, 1, RELU));
    nn_add_layer(&net, conv_layer_alloc(16, 16, 64, 3, 64, 1, same, 2, RELU));
    nn_add_layer(&net, conv_layer_alloc(8, 8, 64, 3, 128, 1, same, 1, RELU));
    nn_add_layer(&net, conv_layer_alloc(8, 8, 128, 3, 128, 1, same, 2, RELU));
    nn_add_layer(&net, dense_layer_alloc(2048, 128, RELU));
    nn_add_layer(&net, dense_layer_alloc(128, 10, SOFTMAX));

    nn_he(net);

    tens inputs = tens_alloc(32, 32, 3);
    void *outputs = NULL;
    mat targets = mat_alloc(10, 1);
    mat grad_in = mat_alloc(10, 1);
    void *grad_out = NULL;
    int prediction;
    int actual;

    for (int i = 0; i < 6; ++i) {
        FILE *f = fopen(files[i], "rb");

        while ((actual = read_next_img(f, inputs)) != -1) {
            nn_forward(net, &inputs, &outputs);
            mat *predicted = (mat *)outputs;

            double max = 0;
            for (int i = 0; i < 10; ++i) {
                if (mat_at(*predicted, i, 0) > max) {
                    max = mat_at(*predicted, i, 0);
                    prediction = i;
                }
            }

            printf("Predicted: %d | Actual: %d\n", prediction, actual);

            mat_fill(targets, 0);
            mat_at(targets, actual, 0) = 1;

            mat_sub(grad_in, *predicted, targets);

            nn_backprop(net, &grad_in, &grad_out, 1e-3);
            tens *input_grad = (tens *)grad_out;

            free(predicted->vals);
            tens_destroy(input_grad);
        }

        fclose(f);
    }

    tens_destroy(&inputs);
    free(targets.vals);
    free(grad_in.vals);

    nn_destroy(net);
}
