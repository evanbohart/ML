#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

typedef float (*func)(float);

float lin(float x);
float dlin(float x);
float sig(float x);
float dsig(float x);
float dtanh(float x);
float relu(float x);
float drelu(float x);
float clip(float x);

float mse(float y, float t);
float cxe(float y, float t);
float dmse(float y, float t);
float dcxe(float y, float t);

typedef int padding_t[4];

enum { TOP, BOTTOM, LEFT, RIGHT };

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
void mat_scale(mat dest, mat m, float a);
void mat_func(mat dest, mat m, func f);
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
void tens3D_scale(tens3D dest, tens3D t, float a);
void tens3D_func(tens3D dest, tens3D t, func f);
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
void tens4D_scale(tens4D dest, tens4D t, float a);
void tens4D_func(tens4D dest, tens4D t, func f);
void tens4D_pad(tens4D dest, tens4D t, padding_t padding);
void tens4D_print(tens4D t);
void tens4D_destroy(tens4D t);
void tens4D_save(tens4D t, FILE *f);
void tens4D_load(tens4D t, FILE *f);

typedef enum layer_type { DENSE, CONV, MAXPOOL, FLATTEN, DENSE_DROPOUT, CONV_DROPOUT,
                          RECURRENT, CONCAT, LSTM, SOFTMAX } layer_type;

typedef struct layer {
    layer_type type;
    void *data;

    void (*forward)(struct layer l, void *x, void **y);
    void (*backprop)(struct layer l, void *dy, void **dx, float rate);
    void (*destroy)(struct layer);

    void (*init)(struct layer);
    void (*print)(struct layer);
    void (*save)(struct layer, FILE *f);
    void (*load)(struct layer, FILE *f);
} layer;

typedef enum { LIN, SIG, TANH, RELU } actfunc;

typedef struct {
    int x_size;
    int y_size;
    int batch_size;
	mat w;
	mat b;
    mat x_cache;
    mat z_cache;
    actfunc activation;
} dense_layer;

layer dense_layer_alloc(int x_size, int y_size,
                        int batch_size, actfunc activation);
void dense_forward(layer l, void *x, void **y);
void dense_backprop(layer l, void *dy, void **dx, float rate);
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
	actfunc activation;
} conv_layer;

layer conv_layer_alloc(int x_rows, int x_cols, int x_depth,
                       int batch_size, int filter_size, int y_depth,
                       int stride, padding_t padding, actfunc activation);
void conv_forward(layer l, void *x, void **y);
void conv_backprop(layer l, void *dy, void **dx, float rate);
void conv_destroy(layer l);
void conv_init(layer l);
void conv_print(layer l);
void conv_save(layer l, FILE *f);
void conv_load(layer l, FILE *f);

typedef struct {
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
void maxpool_forward(layer l, void *x, void **y);
void maxpool_backprop(layer l, void *dy, void **dx, float rate);
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
void flatten_forward(layer l, void *x, void **y);
void flatten_backprop(layer l, void *dy, void **dx, float rate);
void flatten_destroy(layer l);

typedef struct {
    int x_size;
    int batch_size;
    float rate;
    mat mask;
} dense_dropout_layer;

layer dense_dropout_layer_alloc(int x_size, int batch_size, float rate);
void dense_dropout_forward(layer l, void *x, void **y);
void dense_dropout_backprop(layer l, void *dy, void **dx, float rate);
void dense_dropout_destroy(layer l);

typedef struct {
    int x_rows;
    int x_cols;
    int x_depth;
    int batch_size;
    float rate;
    tens4D mask;
} conv_dropout_layer;

layer conv_dropout_layer_alloc(int x_rows, int x_cols, int x_depth,
                               int batch_size, float rate);
void conv_dropout_forward(layer l, void *x, void **y);
void conv_dropout_backprop(layer l, void *dy, void **dx, float rate);
void conv_dropout_destroy(layer l);

typedef struct {
    int x_size;
    int h_size;
    int y_size;
    int batch_size;
    int steps;
    mat w_x;
    mat w_h;
    mat w_y;
    mat b_h;
    mat b_y;
    mat h_0;
    tens3D x_cache;
    tens3D h_z_cache;
    tens3D h_cache;
    tens3D y_z_cache;
    actfunc activation_h;
    actfunc activation_y;
} recurrent_layer;

layer recurrent_layer_alloc(int x_size, int h_size, int y_size,
                            int batch_size, int steps,
                            actfunc activation_h, actfunc activation_y);
void recurrent_forward(layer l, void *x, void **y);
void recurrent_backprop(layer l, void *dy, void **dx, float rate);
void recurrent_destroy(layer l);
void recurrent_init(layer l);
void recurrent_print(layer l);
void recurrent_save(layer l, FILE *f);
void recurrent_load(layer l, FILE *f);

typedef struct {
    int x_size;
    int h_size;
    int y_size;
    int batch_size;
    int steps;
    mat w_x_i;
    mat w_x_f;
    mat w_x_o;
    mat w_x_cc;
    mat w_h_i;
    mat w_h_f;
    mat w_h_o;
    mat w_h_cc;
    mat b_i;
    mat b_f;
    mat b_o;
    mat b_cc;
    mat w_y;
    mat b_y;
    mat h_0;
    mat c_0;
    tens3D x_cache;
    tens3D h_cache;
    tens3D c_cache;
    tens3D i_a_cache;
    tens3D f_a_cache;
    tens3D o_a_cache;
    tens3D cc_a_cache;
    tens3D i_z_cache;
    tens3D f_z_cache;
    tens3D o_z_cache;
    tens3D cc_z_cache;
    tens3D y_z_cache;
    actfunc activation;
} lstm_layer;

layer lstm_layer_alloc(int x_size, int h_size, int y_size,
                       int batch_size, int steps, actfunc activation);
void lstm_forward(layer l, void *x, void **y);
void lstm_backprop(layer l, void *dy, void **dx, float rate);
void lstm_destroy(layer l);
void lstm_init(layer l);
void lstm_print(layer l);
void lstm_save(layer l, FILE *f);
void lstm_load(layer l, FILE *f);

typedef struct concat_layer {
    int x_size;
    int batch_size;
    int y_size;
    int steps;
} concat_layer;

layer concat_layer_alloc(int x_size, int batch_size, int steps);
void concat_forward(layer l, void *x, void **y);
void concat_backprop(layer l, void *dy, void **dx, float rate);
void concat_destroy(layer l);

typedef struct softmax_layer {
    int x_size;
    int batch_size;
    mat y_cache;
} softmax_layer;

layer softmax_layer_alloc(int x_size, int batch_size);
void softmax_forward(layer l, void *x, void **y);
void softmax_backprop(layer l, void *dy, void **dx, float rate);
void softmax_destroy(layer l);

typedef struct {
    int num_layers;
    int max_layers;
    layer *layers;
} nn;

nn nn_alloc(int max_layers);
void nn_add_layer(nn *n, layer l);
void nn_forward(nn n, void *x, void **y);
void nn_backprop(nn n, void *dy, void **dx, float rate);
void nn_destroy(nn n);
void nn_init(nn n);
void nn_print(nn n);
void nn_save(nn n, FILE *f);
void nn_load(nn n, FILE *f);

#ifdef __cplusplus
}
#endif

#endif
