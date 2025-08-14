#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 40
#define BATCHES 10 * 1000 / BATCH_SIZE
#define EPOCHS 10

int read_next_img(FILE *f, tens3D x)
{
    unsigned char target;
    if (!fread(&target, 1, sizeof(target), f)) return -1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                unsigned char val;
                if (!fread(&val, 1, sizeof(val), f)) return -1;
                tens3D_at(x, j, k, i) = val / 255.0f;
            }
        }
    }

    return target;
}


int main(void)
{
    srand(time(0));

    char *files[5];
    get_path(files[0], "cifar-10-batches-bin/data_batch_1.bin");
    get_path(files[1], "cifar-10-batches-bin/data_batch_2.bin");
    get_path(files[2], "cifar-10-batches-bin/data_batch_3.bin");
    get_path(files[3], "cifar-10-batches-bin/data_batch_4.bin");
    get_path(files[4], "cifar-10-batches-bin/data_batch_5.bin");

    nn n = nn_alloc(12);
    nn_add_block(&n, res_block_alloc(2, 32, 32, 3, BATCH_SIZE, 32, 3, 1));
    nn_add_layer(&n, maxpool_layer_alloc(32, 32, 32, BATCH_SIZE, 2));
    nn_add_block(&n, res_block_alloc(2, 16, 16, 32, BATCH_SIZE, 64, 3, 1));
    nn_add_layer(&n, maxpool_layer_alloc(16, 16, 64, BATCH_SIZE, 2));
    nn_add_block(&n, res_block_alloc(2, 8, 8, 64, BATCH_SIZE, 128, 3, 1));
    nn_add_layer(&n, maxpool_layer_alloc(8, 8, 128, BATCH_SIZE, 2));
    nn_add_layer(&n, flatten_layer_alloc(4, 4, 128, BATCH_SIZE));
    nn_add_layer(&n, dense_layer_alloc(2048, 128, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_2D_alloc(128, BATCH_SIZE));
    nn_add_layer(&n, dense_layer_alloc(128, 10, BATCH_SIZE));
    nn_add_layer(&n, softmax_layer_alloc(10, BATCH_SIZE));

    tens x;
    x.type = TENS4D;
    x.t4 = tens4D_alloc(32, 32, 3, BATCH_SIZE);

    tens y;
    y.type = MAT;
    y.m = mat_alloc(10, BATCH_SIZE);

    tens dy;
    dy.type = MAT;
    dy.m = mat_alloc(10, BATCH_SIZE);

    tens dx;
    dx.type = TENS4D;
    dx.t4 = tens4D_alloc(32, 32, 3, BATCH_SIZE);

    mat t = mat_alloc(10, BATCH_SIZE);

    for (int i = 0; i < 5; ++i) {
        FILE *f = fopen(files[i], "rb");

        mat_fill(t, 0.0f);
        for (int j = 0; j < BATCHES; ++j) {
            int target;
            if ((target = read_next_img(f, x.t4.tens3Ds[j])) == -1) {
                exit(EXIT_FAILURE);
            }
            else {
                mat_at(t, target, j) = 1.0f;
            }
        }

        for (int j = 0; j < 100 * 1000; ++j) {
            nn_forward(n, x, &y);

            for (int k = 0; k < 10; ++k) {
                for (int l = 0; l < BATCH_SIZE; ++k) {
                    mat_at(dy.m, k, l) = cxe(mat_at(y.m, k, l), mat_at(t, k, l));
                }
            }

            mat_print(dy.m);

            nn_backprop(n, dy, &dx, 1e-3);
        }
    }

    tens4D_destroy(x.t4);
    free(y.m.vals);
    free(dy.m.vals);
    tens4D_destroy(dx.t4);

    return 0;
}
