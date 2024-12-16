#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>

typedef struct mat {
	int rows;
	int cols;
	double *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

typedef enum actfunc {
    SIGMOID,
    RELU,
    SOFTMAX
} actfunc;

typedef double (*func)(double);

double sig(double x);
double dsig(double x);
double relu(double x);
double drelu(double x);

mat mat_alloc(int rows, int cols);
int mat_compare(mat m1, mat m2);
void mat_rand(mat m, double min, double max);
void mat_normal(mat m, double mean, double stddev);
void mat_zero(mat m);
void mat_copy(mat destination, mat m);
void mat_add(mat destination, mat m1, mat m2);
void mat_sub(mat destination, mat m1, mat m2);
void mat_dot(mat destination, mat m1, mat m2);
void mat_had(mat destination, mat m1, mat m2);
void mat_trans(mat destination, mat m);
void mat_scale(mat destination, mat m, double a);
void mat_func(mat destination, mat m, func f);
void mat_softmax(mat destination, mat m);
void mat_print(mat m);
void mat_load(mat *m, FILE *f);
void mat_save(mat m, FILE *f);

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
void feed_forward(net n, mat inputs);
void backprop(net n, mat inputs, mat targets, double rate);
double mean_squared(double output, double target);
double get_cost(mat outputs, mat targets);
void net_spx(net child1, net child2, net parent1, net parent2);
void net_mutate(net n, double rate, double mean, double stddev);

#endif
