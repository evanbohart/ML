#include "nn.h"
#include <assert.h>
#include <stdbool.h>
#include <string.h>

double rand_normal(double mean, double stddev)
{
	static double n2 = 0;
	static bool cached = false;

	if (!cached) {
		double x;
		double y;
		double r;
		do {
			x = rand_double(-1, 1);
			y = rand_double(-1, 1);
			r = x * x + y * y;
		}
		while (r == 0 || r > 1);

		double n1 = x * sqrt(-2.0 * log(r) / r);
		n2 = y * n1;
		cached = true;
		return n1 * stddev + mean;
	}
	else {
		cached = false;
		return n2 * stddev + mean;
	}
}

void shuffle(void *arr, size_t type_size, int arr_size)
{
	char *temp_arr = (char *) arr;
	
	for (int i = arr_size - 1; i > 0; --i) {
		int j = rand() % (i + 1);
		for (int k = 0; k < (int) type_size; ++k) {
			char temp = temp_arr[i * type_size + k];
			temp_arr[i * type_size + k] = temp_arr[j * type_size + k];
			temp_arr[j * type_size + k] = temp;
		}
	}
}

specimen *gen_alloc(int size, int layers, mat topology)
{
	specimen *gen = malloc(size * sizeof(specimen));
	assert(gen != NULL);
	
	for (int i = 0; i < size; ++i) {
		gen[i].fitness = 0;
		gen[i].n = net_alloc(layers, topology);
	}

	return gen;
}

void gen_destroy(specimen **gen, int size)
{
	for (int i = 0; i < size; ++i) {
		net_destroy(&(*gen)[i].n);
	}

	free(*gen);
	*gen = NULL;
}

void gen_copy(specimen **destination, specimen *gen, int size)
{
	for (int i = 0; i < size; ++i) {
		(*destination)[i].fitness = gen[i].fitness;
		net_copy((*destination)[i].n, gen[i].n);
	}
}

int compare_fitness(const void *p, const void *q)
{
	specimen s1 = *(const specimen *)p;
	specimen s2 = *(const specimen *)q;

	return s2.fitness - s1.fitness;
}

void find_best(specimen *destination, specimen *gen, int new_size, int current_size)
{
	assert(destination[0].n.layers == gen[0].n.layers);
	assert(mat_compare(destination[0].n.topology, gen[0].n.topology));
	assert(current_size % 2 == 0);
	
	qsort(gen, current_size, sizeof(specimen), compare_fitness);
	for (int i = 0; i < new_size; ++i) {
		net_copy(destination[i].n, gen[i].n);
	}
}

void gen_sbx_crossover(specimen *destination, specimen *gen, int size)
{
	assert(destination[0].n.layers == gen[0].n.layers);
	assert(mat_compare(destination[0].n.topology, gen[0].n.topology));
	assert(size % 2 == 0);
	
	for (int i = 0; i < size; i += 2) {
		net_sbx_crossover(destination[i].n, destination[i + 1].n, gen[i].n, gen[i + 1].n);
	}
}

void gen_mutate(specimen *destination, int size, double rate, double mean, double stddev)
{
	for (int i = 0; i < size; ++i) {
		net_mutate(destination[i].n, rate, mean, stddev);
	}
}
