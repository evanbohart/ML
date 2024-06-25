#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>

static inline double rand_double(double min, double max) { return (max - min) * rand() / RAND_MAX + min; }

typedef struct mat {
	int rows;
	int cols;
	double *vals;
} mat;

#define mat_at(m, row, col) ((m).vals[(row) * (m).cols + (col)])

typedef double (*func)(double);

mat mat_alloc(int rows, int cols);
int mat_compare(mat m1, mat m2);
void mat_rand(mat destination, double min, double max);
void mat_copy(mat destination, mat m);
void mat_add(mat destination, mat m1, mat m2);
void mat_sub(mat destination, mat m1, mat m2);
void mat_dot(mat destination, mat m1, mat m2);
void mat_had(mat destination, mat m1, mat m2);
void mat_trans(mat destination, mat m);
void mat_scale(mat destination, mat m, double a);
void mat_func(mat destination, mat m, func f);
void mat_print(mat m);

static inline double sig(double x) { return 1 / (1 + exp(x)); }
static inline double dsig(double x) { return sig(x) * (1 - sig(x)); }
static inline double gaussian(double x) { return exp(-(pow(x, 2))); }

typedef struct net {
	int layers;
	mat topology;
	mat *lins;
	mat *acts;
	mat *weights;
	mat *biases;
} net;

static inline double mean_squared(double output, double target) { return pow(output - target, 2); }

net net_alloc(int layers, mat topology);
void net_destroy(net *n);
void net_copy(net destination, net n);
void net_rand(net n, double min, double max);
void feed_forward(net n, mat inputs, func a);
double get_cost(mat outputs, mat targets);
void net_breed(net destination, net n1, net n2);
void net_mutate(net n, double min, double max);
//double backprop(net n, mat inputs, mat targets, func a, func da, double rate);

#endif

typedef struct specimen {
	double fitness;
	net n;
} specimen;

specimen *gen_alloc(int size, int layers, mat topology);
void gen_destroy(specimen **gen, int size);
void gen_copy(specimen **destination, specimen *gen, int size);
int compare_fitness(const void *p, const void *d);
void find_best(specimen *desintation, specimen *gen, int new_size, int current_size);
void gen_breed(specimen *destination, specimen *gen, int new_size, int current_size);
void gen_mutate(specimen *gen, int size, double min, double max);
