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

enum { R, C, D, B };

typedef struct tens {
    int dims[4];
	float *vals;
} tens;

#define tens_at(t, r, c, d, b) \
    ((t).vals[ \
        ((b) * (t).dims[D] * (t).dims[R] * (t).dims[C]) + \
        ((d) * (t).dims[R] * (t).dims[C]) + \
        ((r) * (t).dims[C]) + \
        (c) \
    ])

tens tens_alloc(int r, int c, int d, int b);

void tens_rand(tens t, float min, float max);
void tens_normal(tens t, float mean, float stddev);
void tens_fill(tens t, float val);
void tens_copy(tens dest, tens t);
void tens_add(tens dest, tens t1, tens t2);
void tens_sub(tens dest, tens t1, tens t2);
void tens_dot(tens dest, tens t1, tens t2);
void tens_had(tens dest, tens t1, tens t2);
void tens_trans(tens dest, tens t, int perm[4]);
void tens_180(tens dest, tens t, int flip[4]);
void tens_scale(tens dest, tens t, float a);
void tens_func(tens dest, tens t, func f);
void tens_softmax(tens dest, tens t);
void tens_pad(tens dest, tens t, int padding[4]);
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
    int x_r;
    int y_r;
    int x_b;
	tens w;
	tens b;
    tens x_reshaped;
    tens dot;
    tens x_reshaped_T;
    tens w_T;
    tens dy_reshaped;
    tens dx_reshaped;
    tens dw;
    tens db;
} dense_layer;

layer dense_layer_alloc(int x_r, int y_r, int x_b);

void dense_forward(layer l, tens x, tens *y);
void dense_backprop(layer l, tens dy, tens *dx, float rate);
void dense_destroy(layer l);

void dense_init(layer l);
void dense_print(layer l);
void dense_save(layer l, FILE *f);
void dense_load(layer l, FILE *f);

enum { TOP, BOTTOM, LEFT, RIGHT };

typedef struct conv_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    int y_r;
    int y_c;
	int w_r;
    int w_c;
    int convolutions;
    int stride;
    int x_padding[4];
    int dy_padding[4];
	tens w;
	tens b;
    tens x_padded;
    tens dy_padded;
    tens w_180;
    tens dw;
    tens db;
} conv_layer;

layer conv_layer_alloc(int x_r, int x_c, int x_d,
                       int x_b, int w_r, int w_c,
                       int convolutions, int stride, int padding[4]);
void conv_forward(layer l, tens x, tens *y);
void conv_backprop(layer l, tens dy, tens *dx, float rate);
void conv_destroy(layer l);

void conv_init(layer l);
void conv_print(layer l);
void conv_save(layer l, FILE *f);
void conv_load(layer l, FILE *f);

typedef struct sig_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens x_cache;
} sig_layer;

layer sig_layer_alloc(int x_r, int x_c,
                      int x_d, int x_b);

void sig_forward(layer l, tens x, tens *y);
void sig_backprop(layer l, tens dy, tens *dx, float rate);
void sig_destroy(layer l);

typedef struct tanh_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens x_cache;
} tanh_layer;

layer tanh_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b);

void tanh_forward(layer l, tens x, tens *y);
void tanh_backprop(layer l, tens dy, tens *dx, float rate);
void tanh_destroy(layer l);

typedef struct relu_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens x_cache;
} relu_layer;

layer relu_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b);

void relu_forward(layer l, tens x, tens *y);
void relu_backprop(layer l, tens dy, tens *dx, float rate);
void relu_destroy(layer l);

typedef struct gelu_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens x_cache;
} gelu_layer;

layer gelu_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b);

void gelu_forward(layer l, tens x, tens *y);
void gelu_backprop(layer l, tens dy, tens *dx, float rate);
void gelu_destroy(layer l);

typedef struct softmax_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens y_cache;
} softmax_layer;

layer softmax_layer_alloc(int x_r, int x_c,
                          int x_d, int x_b);

void softmax_forward(layer l, tens x, tens *y);
void softmax_backprop(layer l, tens dy, tens *dx, float rate);
void softmax_destroy(layer l);

typedef struct maxpool_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    int pooling_r;
    int pooling_c;
    int y_r;
    int y_c;
    tens mask;
} maxpool_layer;

layer maxpool_layer_alloc(int x_r, int x_c, int x_d,
                          int x_b, int pooling_r, int pooling_c);
void maxpool_forward(layer l, tens x, tens *y);
void maxpool_backprop(layer l, tens dy, tens *dx, float rate);
void maxpool_destroy(layer l);

typedef struct reshape_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    int y_r;
    int y_c;
    int y_d;
    int y_b;
} reshape_layer;

layer reshape_layer_alloc(int x_r, int x_c, int x_d,
                          int x_b, int y_r, int y_c,
                          int y_d, int y_b);

void reshape_forward(layer l, tens x, tens *y);
void reshape_backprop(layer l, tens dy, tens *dx, float rate);
void reshape_destroy(layer l);

typedef struct dropout_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    float rate;
    tens mask;
} dropout_layer;

layer dropout_layer_alloc(int x_r, int x_c,
                          int x_d, int x_b, float rate);


void dropout_forward(layer l, tens x, tens *y);
void dropout_backprop(layer l, tens dy, tens *dx, float rate);
void dropout_destroy(layer l);

typedef struct batchnorm_layer {
    int x_r;
    int x_c;
    int x_d;
    int x_b;
    tens gamma;
    tens beta;
    tens var_cache;
    tens z_cache;
    tens dgamma;
    tens dbeta;
} batchnorm_layer;

layer batchnorm_layer_alloc(int x_r, int x_c,
                            int x_d, int x_b);

void batchnorm_forward(layer l, tens x, tens *y);
void batchnorm_backprop(layer l, tens dy, tens *dx, float rate);
void batchnorm_destroy(layer l);

void batchnorm_init(layer l);
void batchnorm_print(layer l);
void batchnorm_save(layer l, FILE *f);
void batchnorm_load(layer l, FILE *f);

typedef struct embedding_layer {
    int x_r;
    int x_b;
    int e_R;
    int v_R;
    tens e;
    tens p;
} embedding_layer;

layer embedding_layer_alloc(int x_r, int x_b, int e_R, int v_R);

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
    int h_R;
    int x_b;
    tens w_q;
    tens w_k;
    tens w_v;
    tens w_o;
    tens alpha_cache;
    tens z_cache;
    tens concat_cache;
} attention_layer;

layer attention_layer_alloc(int seq_len, int d_model,
                            int d_k, int x_b);

void attention_forward(layer l, tens x, tens *y);
void attention_backprop(layer l, tens dy, tens *dx, float rate);
void attention_destroy(layer l);

void attention_init(layer l);
void attention_print(layer l);
void attention_save(layer l, FILE *f);
void attention_load(layer l, FILE *f);

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
