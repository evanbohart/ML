#include <stdio.h>
#include <string.h>
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

void train(cnet cn, net n, char *path)
{
    cnet_he(cn);
    net_he(n);

    char files[5][100];
    get_path(files[0], "cifar-10-batches-bin\\data_batch_1.bin");
    get_path(files[1], "cifar-10-batches-bin\\data_batch_2.bin");
    get_path(files[2], "cifar-10-batches-bin\\data_batch_3.bin");
    get_path(files[3], "cifar-10-batches-bin\\data_batch_4.bin");
    get_path(files[4], "cifar-10-batches-bin\\data_batch_5.bin");

    tens cn_inputs = tens_alloc(32, 32, 3);
    mat n_inputs = mat_alloc(16, 1);
    mat delta = mat_alloc(16, 1);
    mat targets = mat_alloc(10, 1);
    int target_index;
    int epoch = 0;

    for (int i = 0; i < 6; ++i) {
        FILE *f = fopen(files[i], "rb");
        while ((target_index = read_next_img(f, cn_inputs)) != -1) {
            cnet_forward(cn, cn_inputs);
            tens_flatten(n_inputs, cn.acts[cn.layers - 2]);

            mat_fill(targets, 0);
            mat_at(targets, target_index, 0) = 1;

            net_backprop(n, n_inputs, targets, 1e-3, delta);
            cnet_backprop(cn, cn_inputs, delta, 1e-3);

            printf("File %d | Epoch %d\n", i, ++epoch);
        }
    }

    free(targets.vals);
    free(delta.vals);
    free(n_inputs.vals);
    tens_destroy(&cn_inputs);

    FILE *f = fopen(path, "wb");
    cnet_save(cn, f);
    net_save(n, f);
}

void showcase()
{

}

int main(int argc, char **argv)
{
    int cn_layers = 16;
    int cn_filter_size = 3;

    mat cn_convolutions = mat_alloc(cn_layers - 1, 1);
    mat_fill(cn_convolutions, 16);

    mat cn_input_dims = mat_alloc(3, 1);
    mat_at(cn_input_dims, 0, 0) = 32;
    mat_at(cn_input_dims, 1, 0) = 32;
    mat_at(cn_input_dims, 2, 0) = 3;

    cnet cn = cnet_alloc(cn_layers, cn_convolutions, cn_input_dims, cn_filter_size);

    for (int i = 0; i < cn_layers - 1; ++i) {
        cn.actfuncs[i] = RELU;
    }

    int n_layers = 4;

    mat n_topology = mat_alloc(n_layers, 1);
    mat_at(n_topology, 0, 0) = 16;
    mat_at(n_topology, 1, 0) = 64;
    mat_at(n_topology, 2, 0) = 128;
    mat_at(n_topology, 3, 0) = 10;

    net n = net_alloc(n_layers, n_topology);

    for (int i = 0; i < n_layers - 2; ++i) {
        n.actfuncs[i] = RELU;
    }

    n.actfuncs[n_layers - 2] = SOFTMAX;

    char path[100];

    if (argc != 3) {
        printf("Usage: %s <mode> <arguments>\n", argv[0]);
    }
    else if (strcmp(argv[1], "train") == 0) {
        get_path(path, argv[2]);
        train(cn, n, path);
    }
    else if (strcmp(argv[1], "showcase") == 0) {
        get_path(path, argv[2]);
        showcase(cn, n, path);
    }
    else {
        printf("Usage: %s <mode> <arguments>\n", argv[0]);
    }

    free(cn_convolutions.vals);
    free(cn_input_dims.vals);

    cnet_destroy(&cn);
    net_destroy(&n);

    return 0;
}
