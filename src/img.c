#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 40
#define BATCHES 10 * 1000 / BATCH_SIZE
#define EPOCHS 20

int read_next_img(FILE *f, tens3D inputs)
{
    unsigned char target;
    if (!fread(&target, 1, sizeof(target), f)) return -1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                unsigned char val;
                if (!fread(&val, 1, sizeof(val), f)) return -1;
                tens3D_at(inputs, k, j, i) = val / 255.0;
            }
        }
    }

    return target;
}

int showcase(char *path)
{
    char file_path[FILENAME_MAX];
    get_path(file_path, path);

    FILE *net_file = fopen(file_path, "rb");

    nn net = nn_alloc(7);
    padding_t same = { 1, 1, 1, 1 };

    nn_add_layer(&net, conv_layer_alloc(32, 32, 3, BATCH_SIZE, 3, 32, 1, same, 2, RELU));
    nn_add_layer(&net, conv_layer_alloc(16, 16, 32, BATCH_SIZE, 3, 64, 1, same, 2, RELU));
    nn_add_layer(&net, conv_layer_alloc(8, 8, 64, BATCH_SIZE, 3, 64, 1, same, 2, RELU));
    nn_add_layer(&net, conv_layer_alloc(4, 4, 64, BATCH_SIZE, 3, 128, 1, same, 2, RELU));
    nn_add_layer(&net, flatten_layer_alloc(2, 2, 128, BATCH_SIZE));
    nn_add_layer(&net, dense_layer_alloc(512, 128, BATCH_SIZE, RELU));
    nn_add_layer(&net, dense_layer_alloc(128, 10, BATCH_SIZE, SOFTMAX));

    nn_load(net, net_file);

    char showcase_file[FILENAME_MAX];
    get_path(showcase_file, "cifar-10-batches-bin/test_batch.bin");
    FILE *f = fopen(showcase_file, "rb");

    tens4D inputs[BATCHES];
    mat targets[BATCHES];

    for (int i = 0; i < BATCHES; ++i) {
        inputs[i] = tens4D_alloc(32, 32, 3, BATCH_SIZE);
        targets[i] = mat_alloc(10, BATCH_SIZE);
    }

    for (int i = 0; i < BATCHES; ++i) {
        mat_fill(targets[i], 0);
        for (int j = 0; j < BATCH_SIZE; ++j) {
            int target = read_next_img(f, inputs[i].tens3Ds[j]);
            if (target == -1) return EXIT_FAILURE;
            mat_at(targets[i], target, j) = 1;
        }
    }
 
    double correct = 0;

    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int i = 0; i < BATCHES; ++i) {
        void *outputs = NULL;

        nn_forward(net, &inputs[i], &outputs);
        mat *predicted = (mat *)outputs;

        for (int j = 0; j < BATCH_SIZE; ++j) {
            double max = 0;
            int prediction = 0;

            for (int k = 0; k < 10; ++k) {
                if (mat_at(*predicted, k, j) > max) {
                    max = mat_at(*predicted, k, j);
                    prediction = k;
                }
            }

            if (mat_at(targets[i], prediction, j) == 1) ++correct;
        }
    }

    printf("Accuracy: %.2f\n", correct / (BATCHES * BATCH_SIZE));

    for (int i = 0; i < BATCHES; ++i) {
        tens4D_destroy(inputs[i]);
        free(targets[i].vals);
    }

    nn_destroy(net);

    return EXIT_SUCCESS;
}

int train(char *path)
{
    char training_files[5][FILENAME_MAX];
    get_path(training_files[0], "cifar-10-batches-bin/data_batch_1.bin");
    get_path(training_files[1], "cifar-10-batches-bin/data_batch_2.bin");
    get_path(training_files[2], "cifar-10-batches-bin/data_batch_3.bin");
    get_path(training_files[3], "cifar-10-batches-bin/data_batch_4.bin");
    get_path(training_files[4], "cifar-10-batches-bin/data_batch_5.bin");

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

    tens4D inputs[BATCHES];
    mat targets[BATCHES];

    for (int i = 0; i < BATCHES; ++i) {
        inputs[i] = tens4D_alloc(32, 32, 3, BATCH_SIZE);
        targets[i] = mat_alloc(10, BATCH_SIZE);
    }

    mat grad_in = mat_alloc(10, BATCH_SIZE);

    void *outputs;
    void *grad_out;

    for (int i = 0; i < EPOCHS; ++i) {
        for (int j = 0; j < 5; ++j) {
            FILE *f = fopen(training_files[j], "rb");

            for (int k = 0; k < BATCHES; ++k) {
                mat_fill(targets[k], 0);
                for (int l = 0; l < BATCH_SIZE; ++l) {
                    int target = read_next_img(f, inputs[k].tens3Ds[l]);
                    if (target == -1) return EXIT_FAILURE;
                    mat_at(targets[k], target, l) = 1;
                }
            }

            for (int k = 0; k < BATCHES; ++k) {
                outputs = NULL;
                grad_out = NULL;

                nn_forward(net, &inputs[k], &outputs);
                mat *predicted = (mat *)outputs;

                mat_sub(grad_in, *predicted, targets[k]);

                nn_backprop(net, &grad_in, &grad_out, 1e-3);
                tens4D *input_grad = (tens4D *)grad_out;

                free(predicted->vals);
                tens4D_destroy(*input_grad);

                free(outputs);
                free(grad_out);

                printf("here");
            }

            fclose(f);
        }

        printf("Epoch %d complete\n", i + 1);
    }

    for (int i = 0; i < BATCHES; ++i) {
        tens4D_destroy(inputs[i]);
        free(targets[i].vals);
    }

    free(grad_in.vals);

    char file_path[FILENAME_MAX];
    get_path(file_path, path);

    FILE *net_file = fopen(file_path, "wb");

    nn_save(net, net_file);

    nn_destroy(net);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    srand(time(0));

    if (argc != 3) {
        return EXIT_FAILURE;
    }
    else if (strcmp(argv[1], "showcase") == 0) {
        return showcase(argv[2]);
    }
    else if (strcmp(argv[1], "train") == 0) {
        return train(argv[2]);
    }

    return EXIT_FAILURE;
}
