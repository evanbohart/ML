#include "nn.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

mat mat_alloc(int rows, int cols)
{
	mat m;
	m.rows = rows;
	m.cols = cols;
	m.vals = malloc(rows * cols * sizeof(double));
	assert(m.vals != NULL);

	return m;
}

int mat_compare(mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);

	for (int i = 0; i < m1.rows; ++i) {
		for (int j = 0; j < m1.cols; ++j) {
			if (mat_at(m1, i, j) != mat_at(m2, i, j)) {
				return 0;
			}
		}
	}

	return 1;
}

void mat_rand(mat m, double min, double max)
{
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			mat_at(m, i, j) = rand_double(min, max);
		}
	}
}

void mat_zero(mat m)
{
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			mat_at(m, i, j) = 0;
		}
	}
}

void mat_copy(mat destination, mat m)
{
	assert(destination.rows == m.rows);
	assert(destination.cols == m.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = mat_at(m, i, j);
		}
	}
}

void mat_add(mat destination, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(destination.rows == m1.rows);
	assert(destination.cols == m1.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = mat_at(m1, i, j) + mat_at(m2, i, j);
		}
	}
}

void mat_sub(mat destination, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(destination.rows == m1.rows);
	assert(destination.cols == m1.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = mat_at(m1, i, j) - mat_at(m2, i, j);
		}
	}
}

void mat_dot(mat destination, mat m1, mat m2)
{
	assert(m1.cols == m2.rows);
	assert(destination.rows == m1.rows);
	assert(destination.cols == m2.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = 0;
			for (int k = 0; k < m1.cols; ++k) {
				mat_at(destination, i, j) += mat_at(m1, i, k) * mat_at(m2, k, j);
			}
		}
	}
}

void mat_had(mat destination, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(destination.rows == m1.rows);
	assert(destination.cols == m1.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = mat_at(m1, i, j) * mat_at(m2, i, j);
		}
	}
}

void mat_trans(mat destination, mat m)
{
	assert(destination.rows == m.cols);
	assert(destination.cols == m.rows);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = mat_at(m, j, i);
		}
	}
}

void mat_scale(mat destination, mat m, double a)
{
	assert(destination.rows == m.rows);
	assert(destination.cols == m.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = a * mat_at(m, i, j);
		}
	}
}

void mat_func(mat destination, mat m, func f)
{
	assert(destination.rows == m.rows);
	assert(destination.cols == m.cols);

	for (int i = 0; i < destination.rows; ++i) {
		for (int j = 0; j < destination.cols; ++j) {
			mat_at(destination, i, j) = f(mat_at(m, i, j));
		}
	}
}

void mat_print(mat m)
{
  assert(m.vals != NULL);

	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			printf("%f ", mat_at(m, i, j));
		}
		printf("\n");
	}
}

void mat_load(mat *m, FILE **f)
{
    assert(f != NULL);

    fread(&m->rows, sizeof(int), 1, *f);
    fread(&m->cols, sizeof(int), 1, *f);
    fread(m->vals, sizeof(double) * m->rows * m->cols, 1, *f);
}

void mat_save(mat m, FILE *f)
{
    assert(f != NULL);

    fwrite(&m.rows, sizeof(int), 1, f);
    fwrite(&m.cols, sizeof(int), 1, f);
    fwrite(m.vals, sizeof(double) * m.rows * m.cols, 1, f);
    fclose(f);
}
