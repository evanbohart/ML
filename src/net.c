#include "nn.h"
#include "utils.h"
#include <math.h>
#include <assert.h>

double sig(double x) { return 1 / (1 + exp(x)); }

double dsig(double x) { return sig(x) * (1 - sig(x)); }

double relu(double x) { return x * (x > 0); }

double drelu(double x) { return x > 0; }

net net_alloc(int layers, mat topology)
{
	assert(topology.rows == layers);
	assert(topology.cols == 1);
	
    net n;
	n.layers = layers;

	n.topology = mat_alloc(layers, 1);
	mat_copy(n.topology, topology);
	
	n.lins = malloc((layers - 1) * sizeof(mat));
	assert(n.lins);

	for (int i = 0; i < layers - 1; ++i) {
		n.lins[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}

	n.acts = malloc((layers - 1) * sizeof(mat));
	assert(n.acts);

	for (int i = 0; i < layers - 1; ++i) {
		n.acts[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}

	n.weights = malloc((layers - 1) * sizeof(mat));
	assert(n.weights);

	for (int i = 0; i < layers - 1; ++i) {
		n.weights[i] = mat_alloc(mat_at(topology, i + 1, 0), mat_at(topology, i, 0));
	}

	n.biases = malloc((layers - 1) * sizeof(mat));
	assert(n.biases);

	for (int i = 0; i < layers - 1; ++i) {
		n.biases[i] = mat_alloc(mat_at(topology, i + 1, 0), 1);
	}

    n.actfuncs = malloc((layers - 1) * sizeof(actfunc));
    assert(n.actfuncs);

	return n;
}

void net_destroy(net *n) {
	free(n->topology.vals);

	for (int i = 0; i < n->layers - 1; ++i) {
		free(n->lins[i].vals);
		free(n->acts[i].vals);
		free(n->weights[i].vals);
		free(n->biases[i].vals);
	}

	free(n->lins);
	free(n->acts);
	free(n->weights);
	free(n->biases);
    free(n->actfuncs);
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

void net_glorot(net n)
{
    for (int i = 0; i < n.layers - 1; ++i) {
        mat_normal(n.weights[i], 0, 2 / (mat_at(n.topology, i, 0) + mat_at(n.topology, i + 1, 0)));
        mat_fill(n.biases[i], 0);
    }
}

void net_he(net n)
{
    for (int i = 0; i < n.layers - 1; ++i) {
        mat_normal(n.weights[i], 0, 2 / mat_at(n.topology, i, 0));
        mat_fill(n.biases[i], 0);
    }
}

void net_print(net n)
{
    for (int i = 0; i < n.layers - 1; ++i) {
        mat_print(n.weights[i]);
        mat_print(n.biases[i]);
    }
}

void net_forward(net n, mat inputs)
{
	assert(inputs.rows == mat_at(n.topology, 0, 0));
	assert(inputs.cols == 1);

	mat_dot(n.lins[0], n.weights[0], inputs);
	mat_add(n.lins[0], n.lins[0], n.biases[0]);

    switch (n.actfuncs[0]) {
        case SIGMOID:
            mat_func(n.acts[0], n.lins[0], sig);
            break;
        case RELU:
            mat_func(n.acts[0], n.lins[0], relu);
            break;
        case SOFTMAX:
            mat_softmax(n.acts[0], n.lins[0]);
            break;
    }

	for (int i = 1; i < n.layers - 1; ++i) {
		mat_dot(n.lins[i], n.weights[i], n.acts[i - 1]);
		mat_add(n.lins[i], n.lins[i], n.biases[i]);

        switch (n.actfuncs[i]) {
            case SIGMOID:
                mat_func(n.acts[i], n.lins[i], sig);
                break;
            case RELU:
                mat_func(n.acts[i], n.lins[i], relu);
                break;
            case SOFTMAX:
                mat_softmax(n.acts[i], n.lins[i]);
                break;
        }
	}
}

void net_backprop(net n, mat inputs, mat targets, double rate, mat delta)
{
    assert(inputs.rows == mat_at(n.topology, 0, 0));
    assert(inputs.cols == 1);
    assert(targets.rows == mat_at(n.topology, n.layers - 1, 0));
    assert(targets.cols == 1);
    assert(delta.rows == mat_at(n.topology, 0, 0));
    assert(delta.cols == 1);

    net_forward(n, inputs);

    mat *deltas = malloc((n.layers - 1) * sizeof(mat));
    assert(deltas != NULL);

    deltas[n.layers - 2] = mat_alloc(targets.rows, 1);
    mat_sub(deltas[n.layers - 2], n.acts[n.layers - 2], targets);

    for (int i = n.layers - 2; i >= 1; --i) {
        mat weights_trans = mat_alloc(n.weights[i].cols, n.weights[i].rows);
        mat_trans(weights_trans, n.weights[i]);

        mat lins_deriv = mat_alloc(n.lins[i].rows, n.lins[i].cols);

        switch (n.actfuncs[i - 1]) {
            case SIGMOID:
                mat_func(lins_deriv, n.lins[i], dsig);
                break;
            case RELU:
                mat_func(lins_deriv, n.lins[i], drelu);
                break;
            case SOFTMAX:
                break;
        }

        mat had = mat_alloc(lins_deriv.rows, lins_deriv.cols);
        mat_had(had, deltas[i], lins_deriv);

        deltas[i - 1] = mat_alloc(weights_trans.rows, had.cols);
        mat_dot(deltas[i - 1], weights_trans, had);

        free(weights_trans.vals);
        free(lins_deriv.vals);
        free(had.vals);
    }

    mat acts_trans = mat_alloc(inputs.cols, inputs.rows);
    mat_trans(acts_trans, inputs);

    for (int i = 0; i < n.layers - 1; ++i) {
        mat dweight = mat_alloc(n.weights[i].rows, n.weights[i].cols);
        mat_dot(dweight, deltas[i], acts_trans);

        mat dbias = mat_alloc(n.biases[i].rows, n.biases[i].cols);
        mat_copy(dbias, deltas[i]);

        mat_scale(dweight, dweight, rate);
        mat_scale(dbias, dbias, rate);

        mat_sub(n.weights[i], n.weights[i], dweight);
        mat_sub(n.biases[i], n.biases[i], dbias);

        free(acts_trans.vals);
        acts_trans = mat_alloc(n.acts[i].cols, n.acts[i].rows);
        mat_trans(acts_trans, n.acts[i]);

        free(dweight.vals);
        free(dbias.vals);
    }

    free(acts_trans.vals);

    mat weight_trans = mat_alloc(n.weights[0].cols, n.weights[0].rows);
    mat_trans(weight_trans, n.weights[0]);

    mat_dot(delta, weight_trans, deltas[0]);

    free(weight_trans.vals);

    for (int i = 0; i < n.layers - 1; ++i) {
        free(deltas[i].vals);
    }

    free(deltas);
}

void net_spx(net child1, net child2, net parent1, net parent2)
{
	assert(mat_compare(child1.topology, child2.topology));
	assert(mat_compare(child1.topology, parent1.topology));
	assert(mat_compare(child1.topology, parent2.topology));
	

	for (int i = 0; i < child1.layers - 1; ++i) {
		int crossover_point = rand() % (child1.weights[i].rows - 1) + 1;

		for (int j = 0; j < child1.weights[i].rows; ++j) {
			for (int k = 0; k < child1.weights[i].cols; ++k) {
				if (j < crossover_point) {
					mat_at(child1.weights[i], j, k) = mat_at(parent1.weights[i], j, k);
					mat_at(child2.weights[i], j, k) = mat_at(parent2.weights[i], j, k);
				}
				else {
					mat_at(child1.weights[i], j, k) = mat_at(parent2.weights[i], j, k);
					mat_at(child2.weights[i], j, k) = mat_at(parent1.weights[i], j, k);
				}
			}

			if (j < crossover_point) {
				mat_at(child1.biases[i], j, 0) = mat_at(parent1.biases[i], j, 0);
				mat_at(child2.biases[i], j, 0) = mat_at(parent2.biases[i], j, 0);
			}
			else {
				mat_at(child1.biases[i], j, 0) = mat_at(parent2.biases[i], j, 0);
				mat_at(child2.biases[i], j, 0) = mat_at(parent1.biases[i], j, 0);
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
			mat_at(n.biases[i], j, 0) += chance < rate ? rand_normal(mean, stddev) : 0;
		}
	}
}

void net_load(net *n, FILE *f)
{
    for (int i = 0; i < n->layers - 1; ++i) {
        mat_load(&n->lins[i], f);
        mat_load(&n->acts[i], f);
        mat_load(&n->weights[i], f);
        mat_load(&n->biases[i], f);
    }
}

void net_save(net n, FILE *f)
{
    for (int i = 0; i < n.layers - 1; ++i) {
        mat_save(n.lins[i], f);
        mat_save(n.acts[i], f);
        mat_save(n.weights[i], f);
        mat_save(n.biases[i], f);
    }
}
