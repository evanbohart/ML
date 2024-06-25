#include "nn.h"
#include <assert.h>
#include <stdio.h>

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
	for (int i = 0; i < n.layers - 1; ++i) {
		mat_rand(n.weights[i], min, max);
	}

	for (int i = 0; i < n.layers - 1; ++i) {
		mat_rand(n.biases[i], min, max);
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

void net_breed(net destination, net n1, net n2)
{
	assert(destination.layers == n1.layers);
	assert(destination.layers == n2.layers);
	assert(mat_compare(destination.topology, n1.topology));
	assert(mat_compare(destination.topology, n2.topology));

	for (int i = 0; i < destination.layers - 1; ++i) {
		for (int j = 0; j < destination.weights[i].rows; ++j) {
			for (int k = 0; k < destination.weights[i].cols; ++k) {
				mat_at(destination.weights[i], j, k) = (mat_at(n1.weights[i], j, k) + 
							        mat_at(n2.weights[i], j, k)) / 2;
			}
		}

		for (int j = 0; j < destination.biases[i].rows; ++j) {
			mat_at(destination.biases[i], j, 0) = (mat_at(n1.biases[i], j, 0) + 
						       mat_at(n2.biases[i], j, 0)) / 2;
		}
	}
}

void net_mutate(net n, double min, double max)
{
	for (int i = 0; i < n.layers - 1; ++i) {
		for (int j = 0; j < n.weights[i].rows; ++j) {
			for (int k = 0; k < n.weights[i].cols; ++k) {
				mat_at(n.weights[i], j, k) += rand_double(min, max);
			}
		}

		for (int j = 0; j < n.biases[i].rows; ++j) {
			mat_at(n.biases[i], j, 0) += rand_double(min, max);
		}
	}
}

/*
double backprop(net n, mat inputs, mat targets, func a, func da, double rate)
{
	assert(targets.rows == mat_at(n.topology, n.layers - 1, 0));
	assert(targets.cols == pow(2, mat_at(n.topology, 0, 0)));

	feed_forward(n, inputs, a);

	mat *lins_grads = malloc((n.layers - 1) * sizeof(mat));
	assert(lins_grads != NULL);

	lins_grads[n.layers - 2] = mat_alloc(targets.rows, targets.cols);
	mat_sub(lins_grads[n.layers - 2], n.acts[n.layers - 2], targets);

	for (int i = n.layers - 3; i >= 0; --i) {
		mat weights_trans = mat_alloc(n.weights[i + 1].cols, n.weights[i + 1].rows);
		mat_trans(weights_trans, n.weights[i + 1]);
		mat dot = mat_alloc(weights_trans.rows, lins_grads[i + 1].cols);
		mat_dot(dot, weights_trans, lins_grads[i + 1]);
		mat derivs = mat_alloc(n.lins[i].rows, n.lins[i].cols);
		mat_func(derivs, n.lins[i], da);
		lins_grads[i] = mat_alloc(n.lins[i].rows, n.lins[i].cols);
		mat_had(lins_grads[i], dot, derivs);

		free(weights_trans.vals);
		free(dot.vals);
		free(derivs.vals);
	}

	mat *weights_grads = malloc((n.layers - 1) * sizeof(mat));
	assert(weights_grads != NULL);

	for (int i = n.layers - 2; i >= 1; --i) {
		mat acts_trans = mat_alloc(n.acts[i - 1].cols, n.acts[i - 1].rows);
		mat_trans(acts_trans, n.acts[i - 1]);
		weights_grads[i] = mat_alloc(n.weights[i].rows, n.weights[i].cols);
		mat_dot(weights_grads[i], lins_grads[i], acts_trans);
		
		for (int j = 0; j < weights_grads[i].rows; ++j) {
			for (int k = 0; k < weights_grads[i].cols; ++k) {
				mat_at(weights_grads[i], j, k) /= inputs.cols;
			}
		}

		free(acts_trans.vals);
	}

	mat inputs_trans = mat_alloc(inputs.cols, inputs.rows);
	mat_trans(inputs_trans, inputs);
	weights_grads[0] = mat_alloc(lins_grads[0].rows, inputs_trans.cols);
	mat_dot(weights_grads[0], lins_grads[0], inputs_trans);
	
	for (int i = 0; i < weights_grads[0].rows; ++i) {
		for (int j = 0; j < weights_grads[0].cols; ++j) {
			mat_at(weights_grads[0], i, j) /= inputs.cols;
		}
	}

	free(inputs_trans.vals);

	mat *biases_grads = malloc((n.layers - 1) * sizeof(mat));
	assert(biases_grads != NULL);

	for (int i = 0; i < n.layers - 1; ++i) {
		biases_grads[i] = mat_alloc(lins_grads[i].rows, 1);
		for (int j = 0; j < lins_grads[i].rows; ++j) {
			mat_at(biases_grads[i], j, 0) = 0;
			for (int k = 0; k < n.lins[i].cols; ++k) {
				mat_at(biases_grads[i], j, 0) += mat_at(lins_grads[i], j, k);
			}
			mat_at(biases_grads[i], j, 0) /= inputs.cols;
		}
	}

	for (int i = 0; i < n.layers - 1; ++i) {
		mat_scale(weights_grads[i], weights_grads[i], rate);
		mat_scale(biases_grads[i], biases_grads[i], rate);
		mat_sub(n.weights[i], n.weights[i], weights_grads[i]);
		mat_sub(n.biases[i], n.biases[i], biases_grads[i]);

		free(weights_grads[i].vals);
		free(biases_grads[i].vals);
		free(lins_grads[i].vals);
	}
	
	free(lins_grads);
	free(weights_grads);
	free(biases_grads);

	feed_forward(n, inputs, a);
	return get_cost(n.acts[n.layers - 2], targets);
}*/
