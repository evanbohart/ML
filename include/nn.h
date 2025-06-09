#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef float (*func)(float);

float sig(float x);
float dsig(float x);
float relu(float x);
float drelu(float x);
float clip(float x);

typedef int padding_t[4];

enum { TOP, BOTTOM, LEFT, RIGHT };

typedef struct mat {
	int rows;
	int cols;
	float *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

typedef struct tens3D {
	int rows;
	int cols;
	int depth;
	mat *mats;
} tens3D;

#define tens3D_at(t, row, col, depth) (mat_at((t).mats[(depth)], (row), (col)))

typedef struct tens4D {
    int rows;
    int cols;
    int depth;
    int batches;
    tens3D *tens3Ds;
} tens4D;

#define tens4D_at(t, row, col, depth, batch) (tens3D_at((t).tens3Ds[(batch)], (row), (col), (depth)))

mat mat_alloc(int rows, int cols);
void mat_rand(mat m, float min, float max);
void mat_normal(mat m, float mean, float stddev);
void mat_fill(mat m, float val);
void mat_copy(mat destination, mat m);
void mat_add(mat destination, mat m1, mat m2);
void mat_sub(mat destination, mat m1, mat m2);
void mat_dot(mat destination, mat m1, mat m2);
void mat_had(mat destination, mat m1, mat m2);
void mat_trans(mat destination, mat m);
void mat_180(mat destination, mat m);
void mat_scale(mat destination, mat m, float a);
void mat_func(mat destination, mat m, func f);
void mat_softmax(mat destination, mat m);
void mat_pad(mat destination, mat m, padding_t padding);
void mat_convolve(mat destination, mat m, mat filter);
void mat_unflatten(tens4D destination, mat m);
void mat_print(mat m);
void mat_save(mat m, FILE *f);
void mat_load(mat m, FILE *f);

tens3D tens3D_alloc(int rows, int cols, int depth);
void tens3D_rand(tens3D t, float min, float max);
void tens3D_normal(tens3D t, float mean, float stddev);
void tens3D_fill(tens3D t, float val);
void tens3D_copy(tens3D destination, tens3D t);
void tens3D_add(tens3D destination, tens3D t1, tens3D t2);
void tens3D_sub(tens3D destination, tens3D t1, tens3D t2);
void tens3D_had(tens3D destinaiton, tens3D t1, tens3D t2);
void tens3D_trans(tens3D destination, tens3D t);
void tens3D_180(tens3D destination, tens3D t);
void tens3D_scale(tens3D destination, tens3D t, float a);
void tens3D_func(tens3D destination, tens3D t, func f);
void tens3D_pad(tens3D destination, tens3D t, padding_t padding);
void tens3D_flatten(mat destination, tens3D t);
void tens3D_print(tens3D t);
void tens3D_destroy(tens3D t);
void tens3D_save(tens3D t, FILE *f);
void tens3D_load(tens3D t, FILE *f);

tens4D tens4D_alloc(int rows, int cols, int depth, int batches);
void tens4D_rand(tens4D t, float min, float max);
void tens4D_normal(tens4D t, float mean, float stddev);
void tens4D_fill(tens4D t, float val);
void tens4D_copy(tens4D destination, tens4D t);
void tens4D_sub(tens4D destination, tens4D t1, tens4D t2);
void tens4D_had(tens4D destination, tens4D t1, tens4D t2);
void tens4D_trans(tens4D destination, tens4D t);
void tens4D_180(tens4D destination, tens4D t);
void tens4D_scale(tens4D destination, tens4D t, float a);
void tens4D_func(tens4D destination, tens4D t, func f);
void tens4D_pad(tens4D destination, tens4D t, padding_t padding);
void tens4D_flatten(mat destination, tens4D t);
void tens4D_print(tens4D t);
void tens4D_destroy(tens4D t);
void tens4D_save(tens4D t, FILE *f);
void tens4D_load(tens4D t, FILE *f);

typedef enum actfunc { SIGMOID, RELU, SOFTMAX } actfunc;

typedef enum layer_type { DENSE, CONV, MAXPOOL, FLATTEN, DENSE_DROPOUT, CONV_DROPOUT } layer_type;

typedef struct layer {
    layer_type type;
    void *data;

    void (*forward)(struct layer l, void *inputs, void **outputs);
    void (*backprop)(struct layer l, void *grad_in, void **grad_out, float rate);
    void (*destroy)(struct layer);

    void (*he)(struct layer);
    void (*glorot)(struct layer);
    void (*print)(struct layer);
    void (*save)(struct layer, FILE *f);
    void (*load)(struct layer, FILE *f);
} layer;

typedef struct dense_layer {
	mat weights;
	mat biases;
    actfunc activation;
    int input_size;
    int output_size;
    int batch_size;
    mat input_cache;
    mat lins_cache;
} dense_layer;

layer dense_layer_alloc(int input_size, int output_size,
                        int batch_size, actfunc activation);
void dense_forward(layer l, void *inputs, void **outputs);
void dense_backprop(layer l, void *grad_in, void **grad_out, float rate);
void dense_destroy(layer l);
void dense_he(layer l);
void dense_glorot(layer l);
void dense_print(layer l);
void dense_save(layer l, FILE *f);
void dense_load(layer l, FILE *f);

typedef struct conv_layer {
	tens4D filters;
	tens3D biases;
	actfunc activation;
    int input_rows;
    int input_cols;
    int input_channels;
    int batch_size;
    int output_rows;
    int output_cols;
    int convolutions;
	int filter_size;
    int stride;
    padding_t padding;
    tens4D input_cache;
    tens4D lins_cache;
} conv_layer;

layer conv_layer_alloc(int input_rows, int input_cols, int input_channels,
                       int batch_size, int filter_size, int convolutions,
                       int stride, padding_t padding, actfunc activation);
void conv_forward(layer l, void *inputs, void **outputs);
void conv_backprop(layer l, void *grad_in, void **grad_out, float rate);
void conv_destroy(layer l);
void conv_he(layer l);
void conv_glorot(layer l);
void conv_print(layer l);
void conv_save(layer l, FILE *f);
void conv_load(layer l, FILE *f);

typedef struct maxpool_layer {
    int input_rows;
    int input_cols;
    int input_channels;
    int batch_size;
    int pooling_size;
    int output_rows;
    int output_cols;
    tens4D mask;
} maxpool_layer;

layer maxpool_layer_alloc(int input_rows, int input_cols, int input_channels,
                          int batch_size, int pooling_size);
void maxpool_forward(layer l, void *inputs, void **outputs);
void maxpool_backprop(layer l, void *grad_in, void **grad_out, float rate);
void maxpool_destroy(layer l);

typedef struct flatten_layer {
    int input_rows;
    int input_cols;
    int input_channels;
    int batch_size;
    int output_size;
} flatten_layer;

layer flatten_layer_alloc(int input_rows, int input_cols,
                          int input_channels, int batch_size);
void flatten_forward(layer l, void *inputs, void **outputs);
void flatten_backprop(layer l, void *grad_in, void **grad_out, float rate);
void flatten_destroy(layer l);

typedef struct dense_dropout_layer {
    int input_size;
    int batch_size;
    float rate;
    mat mask;
} dense_dropout_layer;

layer dense_dropout_layer_alloc(int input_size, int batch_size, float rate);
void dense_dropout_forward(layer l, void *inputs, void **outputs);
void dense_dropout_backprop(layer l, void *grad_in, void **grad_out, float rate);
void dense_dropout_destroy(layer l);

typedef struct conv_dropout_layer {
    int input_rows;
    int input_cols;
    int input_channels;
    int batch_size;
    float rate;
    tens4D mask;
} conv_dropout_layer;

layer conv_dropout_layer_alloc(int input_rows, int input_cols,
                               int input_channels, int batch_size, float rate);
void conv_dropout_forward(layer l, void *inputs, void **outputs);
void conv_dropout_backprop(layer l, void *grad_in, void **grad_out, float rate);
void conv_dropout_destroy(layer l);

typedef struct nn {
    int num_layers;
    int max_layers;
    layer *layers;
} nn;

nn nn_alloc(int max_layers);
void nn_add_layer(nn *n, layer l);
void nn_forward(nn n, void *inputs, void **outputs);
void nn_backprop(nn n, void *grad_in, void **grad_out, float rate);
void nn_destroy(nn n);
void nn_he(nn n);
void nn_glorot(nn n);
void nn_print(nn n);
void nn_save(nn n, FILE *f);
void nn_load(nn n, FILE *f);

#ifdef __cplusplus
}
#endif

#endif
