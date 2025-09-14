#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <float.h>
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 40
#define BATCHES 10 * 1000 / BATCH_SIZE
#define EPOCHS 19
#define SHOWCASE

int read_next_img(FILE *f, tens x, int batch)
{
    unsigned char target;
    if (!fread(&target, 1, sizeof(target), f)) return -1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                unsigned char val;
                if (!fread(&val, 1, sizeof(val), f)) return -1;
                tens_at(x, j, k, i, batch) = val / 255.0f;
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

    char test_file[FILENAME_MAX];
    get_path(test_file, "cifar-10-batches-bin/test_batch.bin");

    int same[4] = { 1, 1, 1, 1 };

    nn n = nn_alloc(32);

    nn_add_layer(&n, conv_layer_alloc(32, 32, 3, BATCH_SIZE, 3, 3, 32, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(32, 32, 32, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(32, 32, 32, BATCH_SIZE));
    nn_add_layer(&n, conv_layer_alloc(32, 32, 32, BATCH_SIZE, 3, 3, 32, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(32, 32, 32, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(32, 32, 32, BATCH_SIZE));
    nn_add_layer(&n, maxpool_layer_alloc(32, 32, 32, BATCH_SIZE, 2, 2));
#ifdef TRAIN
    nn_add_layer(&n, dropout_layer_alloc(16, 16, 32, BATCH_SIZE, 0.25));
#endif
    nn_add_layer(&n, conv_layer_alloc(16, 16, 32, BATCH_SIZE, 3, 3, 64, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(16, 16, 64, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(16, 16, 64, BATCH_SIZE));
    nn_add_layer(&n, conv_layer_alloc(16, 16, 64, BATCH_SIZE, 3, 3, 64, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(16, 16, 64, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(16, 16, 64, BATCH_SIZE));
    nn_add_layer(&n, maxpool_layer_alloc(16, 16, 64, BATCH_SIZE, 2, 2));
#ifdef TRAIN
    nn_add_layer(&n, dropout_layer_alloc(8, 8, 64, BATCH_SIZE, 0.25));
#endif
    nn_add_layer(&n, conv_layer_alloc(8, 8, 64, BATCH_SIZE, 3, 3, 128, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(8, 8, 128, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(8, 8, 128, BATCH_SIZE));
    nn_add_layer(&n, conv_layer_alloc(8, 8, 128, BATCH_SIZE, 3, 3, 128, 1, same));
    nn_add_layer(&n, batchnorm_layer_alloc(8, 8, 128, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(8, 8, 128, BATCH_SIZE));
    nn_add_layer(&n, maxpool_layer_alloc(8, 8, 128, BATCH_SIZE, 2, 2));
#ifdef TRAIN
    nn_add_layer(&n, dropout_layer_alloc(4, 4, 128, BATCH_SIZE, 0.25));
#endif
    nn_add_layer(&n, reshape_layer_alloc(4, 4, 128, BATCH_SIZE, 2048, BATCH_SIZE, 1, 1));
    nn_add_layer(&n, dense_layer_alloc(2048, 128, BATCH_SIZE));
    nn_add_layer(&n, relu_layer_alloc(128, 1, 1, BATCH_SIZE));
#ifdef TRAIN
    nn_add_layer(&n, dropout_layer_alloc(128, 1, 1, BATCH_SIZE, 0.25));
#endif
    nn_add_layer(&n, dense_layer_alloc(128, 10, BATCH_SIZE));
    nn_add_layer(&n, softmax_layer_alloc(10, 1, 1, BATCH_SIZE));

    char net_file[FILENAME_MAX];
    get_path(net_file, "net.bin");

    FILE *f = fopen(net_file, "rb");
    nn_load(n, f);

    tens x = tens_alloc(32, 32, 3, BATCH_SIZE);
    tens y;

#ifdef TRAIN
    tens dy = tens_alloc(10, 1, 1, BATCH_SIZE);
    tens dx;
#endif

    tens t = tens_alloc(10, 1, 1, BATCH_SIZE);

#ifdef TRAIN
    for (int e = 0; e < EPOCHS; ++e) {
        for (int i = 0; i < 5; ++i) {
            f = fopen(files[i], "rb");

	        for (int b = 0; b < BATCHES; ++b) {
                tens_fill(t, 0.0f);

                for (int j = 0; j < BATCH_SIZE; ++j) {
                    int target;
                    if ((target = read_next_img(f, x, j)) == -1) {
                        exit(EXIT_FAILURE);
                    }
                    else {
                        tens_at(t, target, 0, 0, j) = 1.0f;
                    }
                }

                nn_forward(n, x, &y);

	            for (int k = 0; k < 10; ++k) {
		            for (int l = 0; l < BATCH_SIZE; ++l) {
            		    tens_at(dy, k, 0, 0, l) = dcxe(tens_at(y, k, l), tens_at(t, k, 0, 0, l));
                    }
                }

                nn_backprop(n, dy, &dx, 1e-4);

	            free(y.vals);
	            tens_destroy(dx);

	            printf("EPOCH %d FILE %d BATCH %d\n", e + 1, i + 1, b + 1);
	        }
        }
    }
#endif
#ifdef SHOWCASE
    int correct = 0;
    f = fopen(test_file, "rb");

    for (int i = 0; i < BATCHES; ++i) {
        tens_fill(t, 0.0f);

        for (int j = 0; j < BATCH_SIZE; ++j) {
	        int target;
	        if ((target = read_next_img(f, x, j)) == -1) {
		        exit(EXIT_FAILURE);
            }
	        else {
		        tens_at(t, target, 0, 0, j) = 1.0f;
            }
        }

        nn_forward(n, x, &y);

        for (int j = 0; j < BATCH_SIZE; ++j) {
	        float max = -FLT_MAX;
    	    int index = 0;

            for (int k = 0; k < 10; ++k) {
		        if (tens_at(y, k, 0, 0, j) > max) {
			        max = tens_at(y, k, 0, 0, j);
			        index = k;
		        }
	        }

	        if (tens_at(t, index, 0, 0, j) == 1.0f) {
		        ++correct;
	        }
        }

        tens_destroy(y);
    }

    printf("%f\n", correct / (10 * 1000.0f));
#endif

    tens_destroy(x);

#ifdef TRAIN
    tens_destroy(dy);

    f = fopen(net_file, "wb");
    nn_save(n, f);
#endif
    nn_destroy(n);

    fclose(f);

    return 0;
}
