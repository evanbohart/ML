#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 40

int read_next_img(FILE *f, tens3D inputs)
{
    unsigned char target;
    if (!fread(&target, 1, sizeof(target), f)) return -1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                unsigned char val;
                fread(&val, 1, sizeof(val), f);
                tens3D_at(inputs, k, j, i) = val;
            }
        }
    }

    return target;
}

int main(void)
{
    srand(time(0));

    char files[5][FILENAME_MAX];
    get_path(files[0], "cifar-10-batches-bin/data_batch_1.bin");
    get_path(files[1], "cifar-10-batches-bin/data_batch_2.bin");
    get_path(files[2], "cifar-10-batches-bin/data_batch_3.bin");
    get_path(files[3], "cifar-10-batches-bin/data_batch_4.bin");
    get_path(files[4], "cifar-10-batches-bin/data_batch_5.bin");

    nn net = nn_alloc(12);
    padding_t same = { 1, 1, 1, 1 };

    nn_add_layer(&net, conv_layer_alloc(32, 32, 3, BATCH_SIZE, 3, 32, 1, same, 2, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(16, 16, 32, BATCH_SIZE, 0.2));
    nn_add_layer(&net, conv_layer_alloc(16, 16, 32, BATCH_SIZE, 3, 64, 1, same, 2, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(8, 8, 64, BATCH_SIZE, 0.2));
    nn_add_layer(&net, conv_layer_alloc(8, 8, 64, BATCH_SIZE, 3, 64, 1, same, 2, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(4, 4, 64, BATCH_SIZE, 0.2));
    nn_add_layer(&net, conv_layer_alloc(4, 4, 64, BATCH_SIZE, 3, 128, 1, same, 2, RELU));
    nn_add_layer(&net, conv_dropout_layer_alloc(2, 2, 128, BATCH_SIZE, 0.2));
    nn_add_layer(&net, flatten_layer_alloc(2, 2, 128, BATCH_SIZE));
    nn_add_layer(&net, dense_layer_alloc(512, 128, BATCH_SIZE, RELU));
    nn_add_layer(&net, dense_dropout_layer_alloc(128, BATCH_SIZE, 0.5));
    nn_add_layer(&net, dense_layer_alloc(128, 10, BATCH_SIZE, SOFTMAX));

    nn_he(net);

    tens4D inputs = tens4D_alloc(32, 32, 3, BATCH_SIZE);
    void *outputs;
    mat targets = mat_alloc(10, BATCH_SIZE);
    mat grad_in = mat_alloc(10, BATCH_SIZE);
    void *grad_out;
    int correct;

    for (int i = 0; i < 6; ++i) {
        FILE *f = fopen(files[i], "rb");

        for (int j = 0; j < 10 * 1000 / BATCH_SIZE; ++j) {
            correct = 0;
            mat_fill(targets, 0);
            for (int k = 0; k < BATCH_SIZE; ++k) {
                mat_at(targets, read_next_img(f, inputs.tens3Ds[k]), k) = 1;
            }

            outputs = NULL;
            grad_out = NULL;

            nn_forward(net, &inputs, &outputs);
            mat *predicted = (mat *)outputs;

            for (int k = 0; k < BATCH_SIZE; ++k) {
                double max = 0;
                int prediction = 0;
                for (int l = 0; l < 10; ++l) {
                    if (mat_at(*predicted, l, 0) > max) {
                        max = mat_at(*predicted, l, k);
                        prediction = l;
                    }
                }

                if (mat_at(targets, prediction, k) == 1) {
                    ++correct;
                }
            }

            printf("File %d BATCH %d: %.2f%%\n", i + 1, j + 1, (double)correct / BATCH_SIZE * 100);

            mat_sub(grad_in, *predicted, targets);

            nn_backprop(net, &grad_in, &grad_out, 1e-3);
            tens4D *input_grad = (tens4D *)grad_out;

            free(predicted->vals);
            tens4D_destroy(*input_grad);

            free(outputs);
            free(grad_out);
        }

        fclose(f);
    }

    tens4D_destroy(inputs);
    free(targets.vals);
    free(grad_in.vals);

    nn_destroy(net);
}
