#include "nn.h"
#include <assert.h>

specimen *gen_alloc(int size, int layers, mat topology)
{
	specimen *gen = malloc(size * sizeof(specimen));
	assert(gen != NULL);

	for (int i = 0; i < size; ++i) {
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

	return (s1.fitness < s2.fitness) - (s1.fitness > s2.fitness);
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

void gen_breed(specimen *destination, specimen *gen, int new_size, int current_size)
{
	assert(destination[0].n.layers == gen[0].n.layers);
	assert(mat_compare(destination[0].n.topology, gen[0].n.topology));
	assert(current_size % 2 == 0);
	assert(new_size == current_size / 2);
	
	for (int i = 0; i < new_size; ++i) {
		net_breed(destination[i].n, gen[i * 2].n, gen[i * 2 + 1].n);
	}
}

void gen_mutate(specimen *destination, int size, double min, double max)
{
	for (int i = 0; i < size; ++i) {
		net_mutate(destination[i].n, min, max);
	}
}
