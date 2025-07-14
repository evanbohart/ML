#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 32
#define EPOCHS 1

void read_inputs(mat x_all, FILE *f)
{
    int day_num;
    float profit;
    int day_of_week;
    int month;
    int is_weekend;
    int is_promotion;

    while (
        fread(&day_num, sizeof(day_num), 1, f) &&
        fread(&profit, sizeof(profit), 1, f) && 
        fread(&day_of_week, sizeof(day_of_week), 1, f) &&
        fread(&month, sizeof(month), 1, f) &&
        fread(&is_weekend, sizeof(is_weekend), 1, f) &&
        fread(&is_promotion, sizeof(is_promotion), 1, f)
    ) {
        mat_at(x_all, 0, day_num - 1) = profit;
        mat_at(x_all, day_of_week + 1, day_num - 1) = 1.0f;
        mat_at(x_all, month + 8, day_num - 1) = 1.0f;
        mat_at(x_all, 20, day_num - 1) = is_weekend;
        mat_at(x_all, 21, day_num - 1) = is_promotion;
    }
}

int main(void)
{
    char *file = "profits.bin";
    char path[FILENAME_MAX];

    get_path(path, file);

    FILE *f = fopen(path, "rb");

    mat x_all = mat_alloc(22, 1461);
    mat_fill(x_all, 0.0f);

    read_inputs(x_all, f);

    fclose(f);

    float sum = 0.0f;

    for (int i = 0; i < 1461; ++i) {
        sum += mat_at(x_all, 0, i);
    }

    float mean = sum / 1461;
    float stddev = 0.0f;

    for (int i = 0; i < 1461; ++i) {
        float diff = mat_at(x_all, 0, i) - mean;
        stddev += diff * diff;
    }

    stddev /= 1461;

    for (int i = 0; i < 1461; ++i) {
        mat_at(x_all, 0, i) = (mat_at(x_all, 0, i) - mean) / stddev;
    }

    nn n = nn_alloc(5);

    nn_add_layer(&n, lstm_layer_alloc(22, 64, 1, BATCH_SIZE, 60, LIN));
    nn_add_layer(&n, concat_layer_alloc(1, BATCH_SIZE, 60));
    nn_add_layer(&n, dense_layer_alloc(60, 1, BATCH_SIZE, RELU));

    tens3D x = tens3D_alloc(22, BATCH_SIZE, 60);
    mat targets = mat_alloc(1, BATCH_SIZE);
    mat dy = mat_alloc(1, BATCH_SIZE);

    void *y;
    void *dx;

    for (int i = 0; i < EPOCHS; ++i) {
        for (int j = 0; j < 1461 - BATCH_SIZE - 60; ++j) {
            for (int k = 0; k < BATCH_SIZE; ++k) {
                mat_at(targets, 0, k) = mat_at(x_all, 0, j + k + 60);

                for (int l = 0; l < 22; ++l) {
                    for (int m = 0; m < 60; ++m) {
                        tens3D_at(x, l, k, j) = mat_at(x_all, l, j + m + k);
                    }
                }
            }

            nn_forward(n, &x, &y);

            mat *predicted = (mat *)y;
            mat_sub(dy, *predicted, targets);

            nn_backprop(n, &dy, &dx, 1e-3);

            free(predicted->vals);
            free(((mat *)dx)->vals);
        }
    }

    free(x_all.vals);
    tens3D_destroy(x);

    nn_destroy(n);

    return 0;
}
