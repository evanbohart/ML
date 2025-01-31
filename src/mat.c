#include "nn.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

mat mat_alloc(int rows, int cols)
{
	mat m;

	m.rows = rows;
	m.cols = cols;

	m.vals = malloc(rows * cols * sizeof(double));
	assert(m.vals);

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

void mat_normal(mat m, double mean, double stddev)
{
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            mat_at(m, i, j) = rand_normal(mean, stddev);
        }
    }
}

void mat_fill(mat m, double val)
{
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			mat_at(m, i, j) = val;
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

void mat_softmax(mat destination, mat m)
{
    assert(destination.rows == m.rows);
    assert(destination.cols == m.cols);

    double max = -DBL_MAX;
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            if (mat_at(m, i, j) > max) max = mat_at(m, i, j);
        }
    }

    double sum = 0;
    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            double val = exp(mat_at(m, i, j) - max);
            mat_at(destination, i, j) = val;
            sum += val;
        }
    }

    mat_scale(destination, destination, 1 / sum);
}

void mat_pad(mat destination, mat m)
{
    assert(destination.rows > m.rows);
    assert(destination.cols > m.cols);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            if (i < (destination.rows - m.rows) / 2 || j < (destination.cols - m.cols) / 2 ||
                i > m.rows || j > m.cols) {
                mat_at(destination, i, j) = 0;
            }
            else {
                mat_at(destination, i, j) = mat_at(m, i - (destination.rows - m.rows) / 2,
                                                   j - (destination.rows - m.rows) / 2);
            }
        }
    }
}

void mat_filter(mat destination, mat m, int row, int col)
{
    assert(destination.rows <= m.rows - row);
    assert(destination.cols <= m.cols - col);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
           mat_at(destination, i, j) = mat_at(m, i + row, j + col);
        }
    }
}

void mat_convolve(mat destination, mat m, mat filter)
{
    assert(destination.rows == m.rows - filter.rows + 1);
    assert(destination.cols == m.cols - filter.cols + 1);

    mat patch = mat_alloc(filter.rows, filter.cols);
    mat had = mat_alloc(filter.rows, filter.cols);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            mat_filter(patch, m, i, j);
            mat_had(had, patch, filter);

            double val = 0;
            for (int k = 0; k < had.rows; ++k) {
                for (int l = 0; l < had.cols; ++l) {
                    val += mat_at(had, k, l);
                }
            }

            mat_at(destination, i, j) = val;
        }
    }

    free(patch.vals);
    free(had.vals);
}

double mat_max(mat m)
{
    double max = -DBL_MAX;

    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            if (mat_at(m, i, j) > max) max = mat_at(m, i, j);
        }
    }

    return max;
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

void mat_load(mat *m, FILE *f)
{
    assert(f != NULL);

    fread(&m->rows, sizeof(int), 1, f);
    fread(&m->cols, sizeof(int), 1, f);
    fread(m->vals, sizeof(double), m->rows * m->cols, f);
}

void mat_save(mat m, FILE *f)
{
    assert(f != NULL);

    fwrite(&m.rows, sizeof(int), 1, f);
    fwrite(&m.cols, sizeof(int), 1, f);
    fwrite(m.vals, sizeof(double), m.rows * m.cols, f);
}
