#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef float (*func)(float);

float sig(float x);
float dsig(float x);
float dtanh(float x);
float relu(float x);
float drelu(float x);
float gelu(float x);
float dgelu(float x);
float clip(float x);

float mse(float y, float t);
float cxe(float y, float t);
float dmse(float y, float t);
float dcxe(float y, float t);

typedef int padding_t[4];

enum { TOP, BOTTOM, LEFT, RIGHT };

typedef struct tens {
    int dims;
	int rows;
	int cols;
    int depth;
    int batches;
	float *vals;
} tens;

#define tens2D_at(t, r, c) \
    ((t).vals[ \
        ((r) * (t).cols) + \
        (c) \
    ])

#define tens3D_at(t, r, c, d) \
    ((t).vals[ \
        ((d) * (t).rows * (t).cols) + \
        ((r) * (t).cols) + \
        (c) \
    ])

#define tens4D_at(t, r, c, d, b) \
    ((t).vals[ \
        ((b) * (t).depth * (t).rows * (t).cols) + \
        ((d) * (t).rows * (t).cols) + \
        ((r) * (t).cols) + \
        (c) \
    ])

tens tens2D_alloc(int rows, int cols);
tens tens3D_alloc(int rows, int cols, int depth);
tens tens4D_alloc(int rows, int cols, int depth, int batches);

void tens_rand(tens t, float min, float max);
void tens_normal(tens t, float mean, float stddev);
void tens_fill(tens t, float val);
void tens_copy(tens dest, tens t);
void tens_add(tens dest, tens t1, tens t2);
void tens_sub(tens dest, tens t1, tens t2);
void tens_dot(tens dest, tens t1, tens t2);
void tens_had(tens dest, tens t1, tens t2);
void tens_trans(tens dest, tens t);
void tens_180(tens dest, tens t);
void tens_diag(tens dest, tens t);
void tens_scale(tens dest, tens t, float a);
void tens_func(tens dest, tens t, func f);
void tens_softmax(tens dest, tens t);
void tens_pad(tens dest, tens t, padding_t padding);
void tens_convolve(tens dest, tens t, tens filter);
void tens_print(tens t);
void tens_destroy(tens t);
void tens_save(tens t, FILE *f);
void tens_load(tens t, FILE *f);

typedef struct layer {
    void *data;

    void (*forward)(struct layer l, tens x, tens *y);
    void (*backprop)(struct layer l, tens dy, tens *dx, float rate);
    void (*destroy)(struct layer l);

    void (*init)(struct layer l);
    void (*print)(struct layer l);
    void (*save)(struct layer l, FILE *f);
    void (*load)(struct layer l, FILE *f);
} layer;

typedef struct {
    int x_size;
    int y_size;
    int x_batches;
	tens w;
	tens b;
    tens x_T;
    tens w_T;
    tens dw;
    tens db;
} dense_layer;

layer dense_layer_alloc(int x_size, int y_size, int x_batches);

void dense_forward(layer l, tens x, tens *y);
void dense_backprop(layer l, tens dy, tens *dx, float rate);
void dense_destroy(layer l);

void dense_init(layer l);
void dense_print(layer l);
void dense_save(layer l, FILE *f);
void dense_load(layer l, FILE *f);

typedef struct conv_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    int y_rows;
    int y_cols;
    int convolutions;
	int filter_size;
    int stride;
    padding_t x_padding;
    padding_t dy_padding;
	tens w;
	tens b;
    tens x_padded;
    tens dy_padded;
    tens w_180;
    tens dw;
    tens db;
} conv_layer;

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int x_batches, int filter_size, int convolutions,
                       int stride, padding_t padding);
void conv_forward(layer l, tens x, tens *y);
void conv_backprop(layer l, tens dy, tens *dx, float rate);
void conv_destroy(layer l);

void conv_init(layer l);
void conv_print(layer l);
void conv_save(layer l, FILE *f);
void conv_load(layer l, FILE *f);

typedef struct sig_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens x_cache;
} sig_layer;

layer sig_layer_2D_alloc(int x_size, int x_batches);
layer sig_layer_3D_alloc(int x_rows, int x_cols, int x_batches);
layer sig_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int x_batches);

void sig_forward(layer l, tens x, tens *y);
void sig_backprop(layer l, tens dy, tens *dx, float rate);
void sig_destroy(layer l);

typedef struct tanh_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens x_cache;
} tanh_layer;

layer tanh_layer_2D_alloc(int x_size, int x_batches);
layer tanh_layer_3D_alloc(int x_rows, int x_cols, int x_batches);
layer tanh_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int x_batches);

void tanh_forward(layer l, tens x, tens *y);
void tanh_backprop(layer l, tens dy, tens *dx, float rate);
void tanh_destroy(layer l);

typedef struct relu_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens x_cache;
} relu_layer;

layer relu_layer_2D_alloc(int x_size, int x_batches);
layer relu_layer_3D_alloc(int x_rows, int x_cols, int x_batches);
layer relu_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int x_batches);

void relu_forward(layer l, tens x, tens *y);
void relu_backprop(layer l, tens dy, tens *dx, float rate);
void relu_destroy(layer l);

typedef struct gelu_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens x_cache;
} gelu_layer;

layer gelu_layer_2D_alloc(int x_size, int x_batches);
layer gelu_layer_3D_alloc(int x_rows, int x_cols, int x_batches);
layer gelu_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int x_batches);

void gelu_forward(layer l, tens x, tens *y);
void gelu_backprop(layer l, tens dy, tens *dx, float rate);
void gelu_destroy(layer l);

typedef struct softmax_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens y_cache;
} softmax_layer;

layer softmax_layer_2D_alloc(int x_size, int x_batches);
layer softmax_layer_3D_alloc(int x_rows, int x_cols, int x_batches);
layer softmax_layer_4D_alloc(int x_rows, int x_cols,
                             int x_depth, int x_batches);

void softmax_forward(layer l, tens x, tens *y);
void softmax_backprop(layer l, tens dy, tens *dx, float rate);
void softmax_destroy(layer l);

typedef struct maxpool_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    int y_rows;
    int y_cols;
    int pooling_size;
    tens mask;
} maxpool_layer;

layer maxpool_layer_alloc(int x_rows, int x_cols, int x_depth,
                          int x_batches, int pooling_size);
void maxpool_forward(layer l, tens x, tens *y);
void maxpool_backprop(layer l, tens dy, tens *dx, float rate);
void maxpool_destroy(layer l);

typedef struct reshape_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    int y_rows;
    int y_cols;
    int y_depth;
    int y_batches;
} reshape_layer;

layer reshape_layer_alloc(int x_rows, int x_cols, int x_depth,
                          int x_batches, int y_rows, int y_cols,
                          int y_depth, int y_batches);

void reshape_forward(layer l, tens x, tens *y);
void reshape_backprop(layer l, tens dy, tens *dx, float rate);
void reshape_destroy(layer l);

typedef struct dropout_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    float rate;
    tens mask;
} dropout_layer;

layer dropout_layer_2D_alloc(int x_size, int x_batches, float rate);
layer dropout_layer_3D_alloc(int x_rows, int x_cols,
                             int x_batches, float rate);
layer dropout_layer_4D_alloc(int x_rows, int x_cols,
                             int x_depth, int x_batches, float rate);


void dropout_forward(layer l, tens x, tens *y);
void dropout_backprop(layer l, tens dy, tens *dx, float rate);
void dropout_destroy(layer l);

typedef struct batchnorm_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    tens gamma;
    tens beta;
    tens var_cache;
    tens z_cache;
} batchnorm_layer;

layer batchnorm_layer_2D_alloc(int x_size, int x_batches);
layer batchnorm_layer_4D_alloc(int x_rows, int x_cols,
                               int x_depth, int x_batches);

void batchnorm_2D_forward(layer l, tens x, tens *y);
void batchnorm_2D_backprop(layer l, tens dy, tens *dx, float rate);

void batchnorm_4D_forward(layer l, tens x, tens *y);
void batchnorm_4D_backprop(layer l, tens dy, tens *dx, float rate);

void batchnorm_destroy(layer l);

void batchnorm_init(layer l);
void batchnorm_print(layer l);
void batchnorm_save(layer l, FILE *f);
void batchnorm_load(layer l, FILE *f);

typedef struct embedding_layer {
    int x_size;
    int x_batches;
    int e_size;
    int v_size;
    tens e;
    tens p;
} embedding_layer;

layer embedding_layer_alloc(int x_size, int x_batches, int e_size, int v_size);

void embedding_forward(layer l, tens x, tens *y);
void embedding_backprop(layer l, tens dy, tens *dx, float rate);
void embedding_destroy(layer l);

void embedding_init(layer l);
void embedding_print(layer l);
void embedding_save(layer l, FILE *f);
void embedding_load(layer l, FILE *f);

typedef struct attention_layer {
    int seq_len;
    int d_model;
    int d_k;
    int h_size;
    int x_batches;
    tens w_q;
    tens w_k;
    tens w_v;
    tens w_o;
    tens alpha_cache;
    tens z_cache;
    tens concat_cache;
} attention_layer;

layer attention_layer_alloc(int seq_len, int d_model,
                            int d_k, int x_batches);

void attention_forward(layer l, tens x, tens *y);
void attention_backprop(layer l, tens dy, tens *dx, float rate);
void attention_destroy(layer l);

void attention_init(layer l);
void attention_print(layer l);
void attention_save(layer l, FILE *f);
void attention_load(layer l, FILE *f);

typedef struct block {
    void *data;

    void (*forward)(struct block b, tens x, tens *y);
    void (*backprop)(struct block b, tens dy, tens *dx, float rate);
    void (*destroy)(struct block b);

    void (*init)(struct block b);
    void (*print)(struct block b);
    void (*save)(struct block b, FILE *f);
    void (*load)(struct block b, FILE *f);
} block;

typedef struct res_block {
    int x_rows;
    int x_cols;
    int x_depth;
    int x_batches;
    int convolutions;
    tens skip;
    layer proj_layer;
    layer *conv_layers;
    layer *batchnorm_layers;
    layer *relu_layers;
} res_block;

block res_block_alloc(int x_rows, int x_cols, int x_depth,
                      int x_batches, int convolutions,
                      int filter_size, int stride);
void res_forward(block b, tens x, tens *y);
void res_backprop(block b, tens dy, tens *dx, float rate);
void res_destroy(block b);

void res_init(block b);
void res_print(block b);
void res_save(block b, FILE *f);
void res_load(block b, FILE *f);


typedef struct encoder_block {
    int seq_len;
    int d_model;
    int d_k;
    int h_size;
    int d_ff;
    int x_batches;
    void *attention_layers;
    void *mlp_hidden_layers;
    void *mlp_output_layers;
} encoder_block;

block encoder_block_alloc(int sub_layers, int seq_len, int d_model,
                          int d_k, int d_ff, int x_batches);
void encoder_forward(block eb, tens x, tens *y);
void encoder_backprop(block eb, tens dy, tens *dx);

typedef struct nn {
    int max_layers;
    int num_layers;
    layer *layers;
} nn;

nn nn_alloc(int max_layers);
void nn_add_layer(nn *n, layer l);
void nn_forward(nn n, tens x, tens *y);
void nn_backprop(nn n, tens dy, tens *dx, float rate);
void nn_destroy(nn n);
void nn_init(nn n);
void nn_print(nn n);
void nn_save(nn n, FILE *f);
void nn_load(nn n, FILE *f);

#ifdef __cplusplus
}
#endif

#endif
