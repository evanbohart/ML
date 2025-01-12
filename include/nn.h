#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*func)(double);

double sig(double x);
double dsig(double x);
double relu(double x);
double drelu(double x);

typedef struct mat {
	int rows;
	int cols;
	double *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

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
void mat_pad(mat destination, mat m);
void mat_filter(mat destination, mat m, int row, int col);
void mat_convolve(mat destination, mat m, mat filter);
double mat_max(mat m);
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
void tens_scale(tens destination, tens t, double a);
void tens_func(tens destination, tens t, func f);
void tens_pad(tens destination, tens t);
void tens_filter(tens destination, tens t, int row, int col);
void tens_convolve(mat destination, tens t, tens filter);
void tens_flatten(mat destination, tens t);
void tens_print(tens t);
void tens_destroy(tens *t);
void tens_load(tens *t, FILE *f);
void tens_save(tens t, FILE *f);

typedef enum actfunc {
    SIGMOID,
    RELU,
    SOFTMAX
} actfunc;

typedef struct net {
	int layers;
	mat topology;
	mat *lins;
	mat *acts;
	mat *weights;
	mat *biases;
    actfunc *actfuncs;
} net;

net net_alloc(int layers, mat topology);
void net_destroy(net *n);
void net_copy(net destination, net n);
void net_glorot(net n);
void net_he(net n);
void net_print(net n);
void net_load(net *n, FILE *f);
void net_save(net n, FILE *f);
void net_forward(net n, mat inputs);
void net_backprop(net n, mat inputs, mat targets, double rate, mat delta);

typedef struct cnet {
	int layers;
    mat convolutions;
    mat input_dims;
	tens *lins;
	tens *acts;
	tens **filters;
	tens *biases;
	actfunc *actfuncs;
	int filter_size;
} cnet;

cnet cnet_alloc(int layers, mat convolutions, mat input_dims, int filter_size);
void cnet_destroy(cnet *cn);
void cnet_glorot(cnet cn);
void cnet_he(cnet cn);
void cnet_print(cnet cn);
void cnet_load(cnet *cn, FILE *f);
void cnet_save(cnet cn, FILE *f);
void cnet_forward(cnet cn, tens inputs);
void cnet_backprop(cnet cn, tens inputs, mat delta, double rate);

#ifdef __cplusplus
}
#endif

#endif
