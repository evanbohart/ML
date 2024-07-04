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

typedef double (*func)(double);

double sig(double x);
double dsig(double x);
double relu(double x);
double drelu(double x);

mat mat_alloc(int rows, int cols);
int mat_compare(mat m1, mat m2);
void mat_rand(mat m, double min, double max);
void mat_zero(mat m);
void mat_copy(mat destination, mat m);
void mat_add(mat destination, mat m1, mat m2);
void mat_sub(mat destination, mat m1, mat m2);
void mat_dot(mat destination, mat m1, mat m2);
void mat_had(mat destination, mat m1, mat m2);
void mat_trans(mat destination, mat m);
void mat_scale(mat destination, mat m, double a);
void mat_func(mat destination, mat m, func f);
void mat_print(mat m);
void mat_load(mat *m, FILE **f);
void mat_save(mat m, char *path);

typedef struct net {
	int layers;
	mat topology;
	mat *lins;
	mat *acts;
	mat *weights;
	mat *biases;
} net;

net net_alloc(int layers, mat topology);
void net_destroy(net *n);
void net_copy(net destination, net n);
void net_rand(net n, double min, double max);
void net_rand_weights(net n, double min, double max);
void net_rand_biases(net n, double min, double max);
void net_zero(net n);
void net_load(net *n, FILE **f);
void net_save(net n, char *path);
void feed_forward(net n, mat inputs, func a);
double mean_squared(double output, double target);
double get_cost(mat outputs, mat targets);
void net_spx(net child1, net child2, net parent1, net parent2);
void net_mutate(net n, double rate, double mean, double stddev);

typedef struct specimen {
	double fitness;
	net n;
} specimen;

specimen *gen_alloc(int size, int layers, mat topology);
void gen_destroy(specimen **gen, int size);
void gen_copy(specimen **destination, specimen *gen, int size);
int compare_fitness(const void *p, const void *d);
void find_best(specimen *desintation, specimen *gen, int new_size, int current_size);
void gen_spx(specimen *destination, specimen *gen, int size);
void gen_mutate(specimen *gen, int size, double rate, double mean, double std_dev);

#endif
