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

typedef enum tens_type { MAT, TENS3D, TENS4D } tens_type;

typedef struct {
	int rows;
	int cols;
	float *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

typedef struct {
	int rows;
	int cols;
	int depth;
	mat *mats;
} tens3D;

#define tens3D_at(t, row, col, depth) (mat_at((t).mats[(depth)], (row), (col)))

typedef struct {
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
void mat_copy(mat dest, mat m);
void mat_add(mat dest, mat m1, mat m2);
void mat_sub(mat dest, mat m1, mat m2);
void mat_dot(mat dest, mat m1, mat m2);
void mat_had(mat dest, mat m1, mat m2);
void mat_trans(mat dest, mat m);
void mat_180(mat dest, mat m);
void mat_diag(mat dest, mat m);
void mat_scale(mat dest, mat m, float a);
void mat_func(mat dest, mat m, func f);
void mat_softmax(mat dest, mat m);
void mat_pad(mat dest, mat m, padding_t padding);
void mat_convolve(mat dest, mat m, mat filter);
void mat_print(mat m);
void mat_save(mat m, FILE *f);
void mat_load(mat m, FILE *f);

tens3D tens3D_alloc(int rows, int cols, int depth);
void tens3D_rand(tens3D t, float min, float max);
void tens3D_normal(tens3D t, float mean, float stddev);
void tens3D_fill(tens3D t, float val);
void tens3D_copy(tens3D dest, tens3D t);
void tens3D_add(tens3D dest, tens3D t1, tens3D t2);
void tens3D_sub(tens3D dest, tens3D t1, tens3D t2);
void tens3D_had(tens3D dest, tens3D t1, tens3D t2);
void tens3D_trans(tens3D dest, tens3D t);
void tens3D_180(tens3D dest, tens3D t);
void tens3D_diag(tens3D dest, tens3D t);
void tens3D_scale(tens3D dest, tens3D t, float a);
void tens3D_func(tens3D dest, tens3D t, func f);
void tens3D_softmax(tens3D dest, tens3D t);
void tens3D_pad(tens3D dest, tens3D t, padding_t padding);
void tens3D_print(tens3D t);
void tens3D_destroy(tens3D t);
void tens3D_save(tens3D t, FILE *f);
void tens3D_load(tens3D t, FILE *f);

tens4D tens4D_alloc(int rows, int cols, int depth, int batches);
void tens4D_rand(tens4D t, float min, float max);
void tens4D_normal(tens4D t, float mean, float stddev);
void tens4D_fill(tens4D t, float val);
void tens4D_copy(tens4D dest, tens4D t);
void tens4D_sub(tens4D dest, tens4D t1, tens4D t2);
void tens4D_had(tens4D dest, tens4D t1, tens4D t2);
void tens4D_trans(tens4D dest, tens4D t);
void tens4D_180(tens4D dest, tens4D t);
void tens4D_diag(tens4D dest, tens4D t);
void tens4D_scale(tens4D dest, tens4D t, float a);
void tens4D_func(tens4D dest, tens4D t, func f);
void tens4D_softmax(tens4D dest, tens4D t);
void tens4D_pad(tens4D dest, tens4D t, padding_t padding);
void tens4D_print(tens4D t);
void tens4D_destroy(tens4D t);
void tens4D_save(tens4D t, FILE *f);
void tens4D_load(tens4D t, FILE *f);

typedef struct tens {
    tens_type type;
    union {
        mat m;
        tens3D t3;
        tens4D t4;
    };
} tens;

typedef struct layer {
    void *data;

    void (*forward)(void *l, tens x, tens *y);
    void (*backprop)(void *l, tens dy, tens *dx, float rate);
    void (*destroy)(void *l);

    void (*init)(void *l);
    void (*print)(void *l);
    void (*save)(void *l, FILE *f);
    void (*load)(void *l, FILE *f);
} layer;

typedef struct {
    int x_size;
    int y_size;
    int batch_size;
	mat w;
	mat b;
    mat x_cache;
    mat z_cache;
} dense_layer;

node dense_layer_alloc(int x_size, int y_size, int batch_size);
void dense_forward(layer l, tens x, tens *y);
void dense_backprop(layer l, tens dy, tens *dx, float rate);
void dense_destroy(layer l);
void dense_init(layer l);
void dense_print(layer l);
void dense_save(layer l, FILE *f);
void dense_load(layer l, FILE *f);

typedef struct {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    int y_rows;
    int y_cols;
    int y_depth;
	int filter_size;
    int stride;
    padding_t padding;
	tens4D filters;
	tens3D b;
    tens4D x_cache;
    tens4D z_cache;
} conv_layer;

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int batch_size, int filter_size, int convolutions,
                       int stride, padding_t padding);
void conv_forward(layer l, tens x, tens *y);
void conv_backprop(layer l, tens dy, tens *dx, float rate);
void conv_destroy(layer l);
void conv_init(layer l);
void conv_print(layer l);
void conv_save(layer l, FILE *f);
void conv_load(layer l, FILE *f);

typedef struct sig_layer {
    tens_type x_type;
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    tens x_cache;
} sig_layer;

layer sig_layer_2D_alloc(int x_size, int batch_size);
layer sig_layer_3D_alloc(int x_rows, int x_cols, int batch_size);
layer sig_layer_4D_alloc(int x_rows, int x_cols,
                         int x_depth, int batch_size);

void sig_layer_2D_forward(layer l, tens x, tens *y);
void sig_layer_2D_backprop(layer l, tens dy, tens *dx, float rate);
void sig_layer_2D_destroy(layer l);

void sig_layer_3D_forward(layer l, tens x, tens *y);
void sig_layer_3D_backprop(layer l, tens dy, tens *dx, float rate);
void sig_layer_3D_destroy(layer l);

void sig_layer_4D_forward(layer l, tens x, tens *y);
void sig_layer_4D_backprop(layer l, tens dy, tens *dx, float rate);
void sig_layer_4D_destroy(layer l);

typedef struct tanh_layer {
    tens_type x_type;
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    tens x_cache;
} sig_layer;

layer tanh_layer_2D_alloc(int x_size, int batch_size);
layer tanh_layer_3D_alloc(int x_rows, int x_cols, int batch_size);
layer tanh_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size);

void tanh_layer_2D_forward(layer l, tens x, tens *y);
void tanh_layer_2D_backprop(layer l, tens dy, tens *dx, float rate);
void tanh_layer_2D_destroy(layer l);

void tanh_layer_3D_forward(layer l, tens x, tens *y);
void tanh_layer_3D_backprop(layer l, tens dy, tens *dx, float rate);
void tanh_layer_3D_destroy(layer l);

void tanh_layer_4D_forward(layer l, tens x, tens *y);
void tanh_layer_4D_backprop(layer l, tens dy, tens *dx, float rate);
void tanh_layer_4D_destroy(layer l);

layer relu_layer_2D_alloc(int x_size, int batch_size);
layer relu_layer_3D_alloc(int x_rows, int x_cols, int batch_size);
layer relu_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size);

void relu_layer_2D_forward(layer l, tens x, tens *y);
void relu_layer_2D_backprop(layer l, tens dy, tens *dx, float rate);
void relu_layer_2D_destroy(layer l);

void relu_layer_3D_forward(layer l, tens x, tens *y);
void relu_layer_3D_backprop(layer l, tens dy, tens *dx, float rate);
void relu_layer_3D_destroy(layer l);

void relu_layer_4D_forward(layer l, tens x, tens *y);
void relu_layer_4D_backprop(layer l, tens dy, tens *dx, float rate);
void relu_layer_4D_destroy(layer l);

layer gelu_layer_2D_alloc(int x_size, int batch_size);
layer gelu_layer_3D_alloc(int x_rows, int x_cols, int batch_size);
layer gelu_layer_4D_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size);

void gelu_layer_2D_forward(layer l, tens x, tens *y);
void gelu_layer_2D_backprop(layer l, tens dy, tens *dx, float rate);
void gelu_layer_2D_destroy(layer l);

void gelu_layer_3D_forward(layer l, tens x, tens *y);
void gelu_layer_3D_backprop(layer l, tens dy, tens *dx, float rate);
void gelu_layer_3D_destroy(layer l);

void gelu_layer_4D_forward(layer l, tens x, tens *y);
void gelu_layer_4D_backprop(layer l, tens dy, tens *dx, float rate);
void gelu_layer_4D_destroy(layer l);

typedef struct maxpool_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    int pooling_size;
    int y_rows;
    int y_cols;
    tens4D mask;
} maxpool_layer;

layer maxpool_layer_alloc(int x_rows, int x_cols, int x_depth,
                          int batch_size, int pooling_size);
void maxpool_forward(layer l, tens x, tens *y);
void maxpool_backprop(layer l, tens dy, tens *dx, float rate);
void maxpool_destroy(layer l);

typedef struct {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    int y_size;
} flatten_layer;

layer flatten_layer_alloc(int x_rows, int x_cols,
                          int x_depth, int batch_size);
void flatten_forward(layer l, tens x, tens *y);
void flatten_backprop(layer l, tens dy, tens *dx, float rate);
void flatten_destroy(layer l);

typedef struct dropout_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    float rate;
    tens_type x_type;
    tens mask;
} dropout_layer;

layer dropout_layer_alloc(tens_type x_type, int x_rows, int x_cols,
                          int x_depth, int batch_size, float rate);
void dropout_forward(layer l, tens x, tens *y);
void dropout_backprop(layer l, tens dy, tens *dx, float rate);
void dropout_destroy(layer l);

typedef struct batchnorm_layer {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    tens_type x_type;
    mat gamma;
    mat beta;
    mat var_cache;
    tens z_cache;
} batchnorm_layer;

layer batchnorm_layer_2D_alloc(int x_size, int batch_size);
layer batchnorm_layer_4D_alloc(int x_rows, int x_cols,
                               int x_depth, int batch_size);
void batchnorm_2D_forward(layer l, tens x, tens *y);
void batchnorm_2D_backprop(layer l, tens dy, tens *dx, float rate);
void batchnorm_2D_destroy(layer l);
void batchnorm_4D_forward(layer l, tens x, tens *y);
void batchnorm_4D_backprop(layer l, tens dy, tens *dx, float rate);
void batchnorm_4D_destroy(layer l);

void batchnorm_init(layer l);
void batchnorm_print(layer l);
void batchnorm_save(layer l, FILE *f);
void batchnorm_load(layer l, FILE *f);

typedef struct embedding_layer {
    int x_size;
    int batch_size;
    int e_size;
    int v_size;
    mat e;
    mat p;
} embedding_layer;

layer embedding_layer_alloc(int x_size, int batch_size, int e_size, int v_size);

void embedding_forward(layer l, tens x, tens *y);
void embedding_backprop(layer l, tens dy, tens *dx, float rate);
void embedding_destroy(layer l);

void embedding_init(layer l, init_type type);
void embedding_print(layer l);
void embedding_save(layer l, FILE *f);
void embedding_load(layer l, FILE *f);

typedef struct attention_layer {
    int x_size;
    int e_size;
    int h_size;
    int batch_size;
    int qk_size;
    tens3D w_q;
    tens3D w_k;
    tens3D w_v;
    mat w_o;
} attention_layer;

layer attention_layer_alloc(int x_size, int e_size, int h_size, int batch_size);

void attention_forward(layer l, tens x, tens *y);
void attention_backprop(layer l, tens dy, tens *dx, float rate);
void attention_destroy(layer l);

void attention_init(layer l, init_type type);
void attention_print(layer l);
void attention_save(layer l, FILE *f);
void attention_load(layer l, FILE *f);

typedef enum block_type { RES, ENCODER } block_type;

typedef struct block {
    void *data;

    void (*forward)(void *b, tens x, tens *y);
    void (*backprop)(void *b, tens dy, tens *dx, float rate);
    void (*destroy)(void *b);

    void (*init)(void *b);
    void (*print)(void *b);
    void (*save)(void *b, FILE *f);
    void (*load)(void *b, FILE *f);
} block;

typedef struct res_block {
    int sub_layers;
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    layer *conv_layers;
    layer *batchnorm_layers;
    layer *relu_layers;
} res_block;

block res_block_alloc(int sub_layers, int x_rows, int x_cols,
                      int x_depth, int batch_size, int filter_size,
                      int y_depth, int stride, padding_t padding);
void res_forward(block rb, tens x, tens *y);
void res_backprop(block rb, tens dy, tens *dx, float rate);
void res_destroy(block rb);

typedef struct encoder_block {
    int x_size;
    int e_size;
    int h_size;
    int qk_size;
    int m_size;
    int batch_size;
    layer *layers;
} encoder_block;

block encoder_block_alloc(int sub_layres, int x_size, int e_size,
                          int h_size, int m_size, int batch_size);
void encoder_forward(block eb, tens x, tens *y);
void encoder_backprop(block eb, tens dy, tens *dx);

typedef enum node_type { LAYER, BLOCK } node_type;

typedef struct node {
    node_type type;

    union {
        layer l;
        block b;
    };

    struct node *next;
    struct node *prev;
} node;

typedef struct nn {
    node *head;
    node *tail;
} nn;

nn nn_init(void);
void nn_add_layer(nn *n, layer l);
void nn_add_block(nn *n, block b);
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
