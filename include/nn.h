#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef double (*func)(double);

double sig(double x);
double dsig(double x);
double relu(double x);
double drelu(double x);
double clip(double x);

typedef struct mat {
	int rows;
	int cols;
	double *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

typedef int padding_t[4];

enum { TOP, BOTTOM, LEFT, RIGHT };

mat mat_alloc(int rows, int cols);
int mat_compare(mat m1, mat m2);
void mat_rand(mat m, double min, double max);
void mat_normal(mat m, double mean, double stddev);
void mat_fill(mat m, double val);
void mat_copy(mat destination, mat m);
void mat_add(mat destination, mat m1, mat m2);
void mat_sub(mat destination, mat m1, mat m2);
void mat_dot(mat destination, mat m1, mat m2);
void mat_had(mat destination, mat m1, mat m2);
void mat_trans(mat destination, mat m);
void mat_scale(mat destination, mat m, double a);
void mat_func(mat destination, mat m, func f);
void mat_softmax(mat destination, mat m);
void mat_pad(mat destination, mat m, padding_t padding);
void mat_filter(mat destination, mat m, int row, int col);
void mat_convolve(mat destination, mat m, mat filter);
void mat_maxpool(mat destination, mat mask, mat m, int pooling_size);
void mat_maxunpool(mat destination, mat mask, mat m, int pooling_size);
void mat_print(mat m);
void mat_load(mat *m, FILE *f);
void mat_save(mat m, FILE *f);

typedef struct tens {
	int rows;
	int cols;
	int depth;
	mat *mats;
} tens;

#define tens_at(t, row, col, depth) (mat_at((t).mats[(depth)], (row), (col)))

tens tens_alloc(int rows, int cols, int depth);
void tens_rand(tens t, double min, double max);
void tens_normal(tens t, double mean, double stddev);
void tens_fill(tens t, double val);
void tens_copy(tens destination, tens t);
void tens_add(tens destination, tens t1, tens t2);
void tens_sub(tens destination, tens t1, tens t2);
void tens_had(tens destinaiton, tens t1, tens t2);
void tens_scale(tens destination, tens t, double a);
void tens_func(tens destination, tens t, func f);
void tens_pad(tens destination, tens t, padding_t padding);
void tens_filter(tens destination, tens t, int row, int col);
void tens_maxpool(tens destination, tens mask, tens t, int pooling_size);
void tens_avgpool(tens destination, tens mask, tens t, int pooling_size);
void tens_maxunpool(tens destination, tens mask, tens t, int pooling_size);
void tens_flatten(mat destination, tens t);
void tens_print(tens t);
void tens_destroy(tens *t);
void tens_load(tens *t, FILE *f);
void tens_save(tens t, FILE *f);

typedef enum actfunc { SIGMOID, RELU, SOFTMAX } actfunc;

typedef enum layer_type { DENSE, CONV } layer_type;

typedef struct layer {
    layer_type type;
    void *data;

    void (*forward)(struct layer l, void *inputs, void **outputs);
    void (*backprop)(struct layer l, void *grad_in, void **grad_out, double rate);
    void (*destroy)(struct layer);
} layer;

typedef struct dense_layer {
	mat weights;
	mat biases;
    actfunc activation;
    int input_size;
    int output_size;
    mat input_cache;
    mat lins_cache;
} dense_layer;

layer dense_layer_alloc(int input_size, int output_size, actfunc activation);
void dense_forward(layer l, void *inputs, void **outputs);
void dense_backprop(layer l, void *grad_in, void **grad_out, double rate);
void dense_destroy(layer l);

typedef struct conv_layer {
	tens *filters;
	tens biases;
	actfunc activation;
    int input_rows;
    int input_cols;
    int input_channels;
    int conv_rows;
    int conv_cols;
    int output_rows;
    int output_cols;
    int output_channels;
    int convolutions;
	int filter_size;
    int stride;
    padding_t padding;
    int pooling_size;
    tens input_cache;
    tens lins_cache;
    tens pooling_mask;
} conv_layer;

layer conv_layer_alloc(int input_rows, int input_cols, int input_channels,
                       int filter_size, int convolutions, int stride,
                       padding_t padding, int pooling_size, actfunc activation);
void conv_forward(layer l, void *inputs, void **outputs);
void conv_backprop(layer l, void *grad_in, void **grad_out, double rate);
void conv_destroy(layer l);

typedef struct nn {
    int num_layers;
    int max_layers;
    layer *layers;
} nn;

nn nn_alloc(int max_layers);
void nn_forward(nn n, void *inputs, void **outputs);
void nn_backprop(nn n, void *grad_in, void **grad_out, double rate);
void nn_destroy(nn n);

#ifdef __cplusplus
}
#endif

#endif
