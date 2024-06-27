#include "nn.h"
#include <assert.h>
#include <stdio.h>

double sig(double x) { return 1 / (1 + exp(x)); }

double dsig(double x) { return sig(x) * (1 - sig(x)); }

double relu(double x) { return x * (x > 0); }

double drelu(double x) { return x > 0; }

net net_alloc(int layers, mat topology)
{
	net n;
	n.layers = layers;

	assert(topology.rows == layers);
	assert(topology.cols == 1);
	
	n.topology = mat_alloc(layers, 1);
	mat_copy(n.topology, topology);
	
	n.lins = malloc((layers - 1) * sizeof(mat));
	assert(n.lins != NULL);

	for (int i = 0; i < layers - 1; ++i) {
		n.lins[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}
	n.acts = malloc((layers - 1) * sizeof(mat));
	assert(n.acts != NULL);

	for (int i = 0; i < layers - 1; ++i) {
		n.acts[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}	

	n.weights = malloc((layers - 1) * sizeof(mat));
	assert(n.weights != NULL);

	for (int i = 0; i < layers - 1; ++i) {
		n.weights[i] = mat_alloc(mat_at(topology, i + 1, 0), mat_at(topology, i, 0));
	}

	n.biases = malloc((layers - 1) * sizeof(mat));
	assert(n.biases != NULL);

	for (int i = 0; i < layers - 1; ++i) {
		n.biases[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}

	return n;
}

void net_destroy(net *n) {
	free(n->topology.vals);
	n->topology.vals = NULL;

	for (int i = 0; i < n->layers - 1; ++i) {
		free(n->lins[i].vals);
		n->lins[i].vals = NULL;
		free(n->acts[i].vals);
		n->acts[i].vals = NULL;
		free(n->weights[i].vals);
		n->weights[i].vals = NULL;
		free(n->biases[i].vals);
		n->biases[i].vals = NULL;
	}

	free(n->lins);
	n->lins = NULL;
	free(n->acts);
	n->acts = NULL;
	free(n->weights);
	n->weights = NULL;
	free(n->biases);
	n->biases = NULL;
}

void net_copy(net destination, net n)
{
	assert(destination.layers == n.layers);
	assert(mat_compare(destination.topology, n.topology));

	for (int i = 0; i < destination.layers - 1; ++i) {
		mat_copy(destination.lins[i], n.lins[i]);
		mat_copy(destination.acts[i], n.acts[i]);
		mat_copy(destination.weights[i], n.weights[i]);
		mat_copy(destination.biases[i], n.biases[i]);
	}
}

void net_rand(net n, double min, double max)
{
	net_rand_weights(n, min, max);
	net_rand_biases(n, min, max);
}

void net_rand_weights(net n, double min, double max)
{
	for (int i = 0; i < n.layers - 1; ++i) {
		mat_rand(n.weights[i], min, max);
	}
}

void net_rand_biases(net n, double min, double max)
{
	for (int i = 0; i < n.layers - 1; ++i) {
		mat_rand(n.biases[i], min, max);
	}
}

void net_zero(net n) {
	for (int i = 0; i < n.layers - 1; ++i) {
		mat_zero(n.weights[i]);
	}

	for (int i = 0; i < n.layers - 1; ++i) {
		mat_zero(n.biases[i]);
	}
}

void feed_forward(net n, mat inputs, func a)
{
	assert(inputs.rows == mat_at(n.topology, 0, 0));
	assert(inputs.cols == 1);

	mat_dot(n.lins[0], n.weights[0], inputs);
	mat_add(n.lins[0], n.lins[0], n.biases[0]);
	mat_func(n.acts[0], n.lins[0], a);

	for (int i = 1; i < n.layers - 1; ++i) {
		mat_dot(n.lins[i], n.weights[i], n.acts[i - 1]);
		mat_add(n.lins[i], n.lins[i], n.biases[i]);
		mat_func(n.acts[i], n.lins[i], a);
	}
}

double mean_squared(double output, double target) { return pow(output - target, 2); }

double get_cost(mat outputs, mat targets)
{
	assert(outputs.rows == targets.rows);
	assert(outputs.cols == targets.cols);

	double cost = 0.0;

	for (int i = 0; i < outputs.rows; ++i) {
		for (int j = 0; j < outputs.cols; ++j) {
			cost += mean_squared(mat_at(outputs, i, j), mat_at(targets, i, j));
		}
	}

	return cost / (outputs.rows * outputs.cols);
}

void net_sbx_crossover(net destination1, net destination2, net n1, net n2)
{
	assert(destination1.layers == destination2.layers);
	assert(destination1.layers == n1.layers);
	assert(destination1.layers == n2.layers);
	assert(mat_compare(destination1.topology, destination2.topology));
	assert(mat_compare(destination1.topology, n1.topology));
	assert(mat_compare(destination1.topology, n2.topology));

	for (int i = 0; i < destination1.layers - 1; ++i) {
		int crossover_point = rand() % destination1.weights[i].rows;

		for (int j = 0; j < destination1.weights[i].rows; ++j) {
			for (int k = 0; k < destination1.weights[i].cols; ++k) {
				if (j < crossover_point) {
					mat_at(destination1.weights[i], j, k) = mat_at(n1.weights[i], j, k);
					mat_at(destination2.weights[i], j, k) = mat_at(n2.weights[i], j, k);
				}
				else {
					mat_at(destination1.weights[i], j, k) = mat_at(n2.weights[i], j, k);
					mat_at(destination2.weights[i], j, k) = mat_at(n1.weights[i], j, k);
				}
			}

			if (j < crossover_point) {
				mat_at(destination1.biases[i], j, 1) = mat_at(n1.biases[i], j, 1);
				mat_at(destination2.biases[i], j, 1) = mat_at(n2.biases[i], j, 1);
			}
			else {
				mat_at(destination1.biases[i], j, 1) = mat_at(n2.biases[i], j, 1);
				mat_at(destination2.biases[i], j, 1) = mat_at(n1.biases[i], j, 1);
			}
		}
	}
}

void net_mutate(net n, double rate, double mean, double stddev)
{
	for (int i = 0; i < n.layers - 1; ++i) {
		for (int j = 0; j < n.weights[i].rows; ++j) {
			for (int k = 0; k < n.weights[i].cols; ++k) {
				double chance = rand_double(0, 1);
				mat_at(n.weights[i], j, k) += chance < rate ? rand_normal(mean, stddev) : 0;			
			}
		}

		for (int j = 0; j < n.biases[i].rows; ++j) {
			double chance = rand_double(0, 1);
			mat_at(n.biases[i], j, 1) += chance < rate ? rand_normal(mean, stddev) : 0;
		}
	}
}
