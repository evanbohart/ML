#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "nn.h"
#include "utils.h"

#define DAYS 737

#define BATCH_SIZE 32
#define EPOCHS 1

void read_inputs(mat x_all, FILE *f)
{
    int day_num;
    float sales;
    float avg_sales;
    int day_of_week;
    int month;
    int is_weekend;
    int is_promotion;
    int is_holiday;

    while (
        fread(&day_num, sizeof(day_num), 1, f) &&
        fread(&sales, sizeof(sales), 1, f) &&
        fread(&avg_sales, sizeof(avg_sales), 1, f) &&
        fread(&day_of_week, sizeof(day_of_week), 1, f) &&
        fread(&month, sizeof(month), 1, f) &&
        fread(&is_weekend, sizeof(is_weekend), 1, f) &&
        fread(&is_promotion, sizeof(is_promotion), 1, f) &&
        fread(&is_holiday, sizeof(is_holiday), 1, f)
    ) {
        mat_at(x_all, 0, day_num - 1) = sales;
        mat_at(x_all, 1, day_num - 1) = avg_sales;
        mat_at(x_all, day_of_week + 2, day_num - 1) = 1.0f;
        mat_at(x_all, month + 9, day_num - 1) = 1.0f;
        mat_at(x_all, 21, day_num - 1) = is_weekend;
        mat_at(x_all, 22, day_num - 1) = is_promotion;
        mat_at(x_all, 23, day_num - 1) = is_holiday;
    }
}

int main(void)
{
    char *file = "sales.bin";
    char path[FILENAME_MAX];

    get_path(path, file);

    FILE *f = fopen(path, "rb");

    mat x_all = mat_alloc(24, DAYS);
    mat_fill(x_all, 0.0f);

    read_inputs(x_all, f);

    fclose(f);

    float min_raw = FLT_MAX;
    float max_raw = -FLT_MAX;
    float min_avg = FLT_MAX;
    float max_avg = -FLT_MAX;

    for (int i = 0; i < DAYS; ++i) {
        min_raw = fmin(min_raw, mat_at(x_all, 0, i));
        max_raw = fmax(max_raw, mat_at(x_all, 0, i));
        min_avg = fmin(min_avg, mat_at(x_all, 1, i));
        max_avg = fmax(max_avg, mat_at(x_all, 1, i));
    }

    for (int i = 0; i < DAYS; ++i) {
        mat_at(x_all, 0, i) = (mat_at(x_all, 0, i) - min_raw) / (max_raw - min_raw);
        mat_at(x_all, 1, i) = (mat_at(x_all, 1, i) - min_avg) / (max_avg - min_avg);
    }

    mat_print(x_all);

        return 0;

    nn n = nn_alloc(5);

    nn_add_layer(&n, lstm_layer_alloc(24, 64, 1, BATCH_SIZE, 60, LIN));
    nn_add_layer(&n, concat_layer_alloc(1, BATCH_SIZE, 60));
    nn_add_layer(&n, dense_layer_alloc(60, 1, BATCH_SIZE, RELU));

    tens3D x = tens3D_alloc(24, BATCH_SIZE, 60);
    mat targets = mat_alloc(1, BATCH_SIZE);
    mat dy = mat_alloc(1, BATCH_SIZE);

    void *y;
    void *dx;

    for (int i = 0; i < EPOCHS; ++i) {
        for (int j = 0; j < DAYS - BATCH_SIZE - 60; ++j) {
            for (int k = 0; k < BATCH_SIZE; ++k) {
                mat_at(targets, 0, k) = mat_at(x_all, 0, j + k + 60);

                for (int l = 0; l < 24; ++l) {
                    for (int m = 0; m < 60; ++m) {
                        tens3D_at(x, l, k, j) = mat_at(x_all, l, j + m + k);
                    }
                }
            }

            nn_forward(n, &x, &y);

            mat *predicted = (mat *)y;

            for (int i = 0; i < BATCH_SIZE; ++i) {
                mat_at(dy, 0, i) = dmse(mat_at(*predicted, 0, i), mat_at(targets, 0, i));
            }

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
