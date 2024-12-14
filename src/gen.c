#include "snake.h"
#include "utils.h"
#include <assert.h>
#include <stdbool.h>
#include <string.h>

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
	assert(current_size % new_size == 0);
	
	qsort(gen, current_size, sizeof(specimen), compare_fitness);
	for (int i = 0; i < new_size; ++i) {
		destination[i].fitness = gen[i].fitness;
		net_copy(destination[i].n, gen[i].n);
	}
}

void gen_spx(specimen *destination, specimen *gen, int size)
{
	assert(destination[0].n.layers == gen[0].n.layers);
	assert(mat_compare(destination[0].n.topology, gen[0].n.topology));
	assert(size % 2 == 0);
	
	for (int i = 0; i < size; i += 2) {
		net_spx(destination[i].n, destination[i + 1].n, gen[i].n, gen[i + 1].n);
	}
}

void gen_mutate(specimen *gen, int size, double rate, double mean, double stddev)
{
	for (int i = 0; i < size; ++i) {
		net_mutate(gen[i].n, rate, mean, stddev);
	}
}
