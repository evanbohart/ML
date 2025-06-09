#include <assert.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

mat mat_alloc(int rows, int cols)
{
	mat m;
	m.rows = rows;
	m.cols = cols;
	m.vals = malloc(rows * cols * sizeof(float));

	return m;
}

void mat_rand(mat m, float min, float max)
{
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			mat_at(m, i, j) = rand_float(min, max);
		}
	}
}

void mat_normal(mat m, float mean, float stddev)
{
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            mat_at(m, i, j) = rand_normal(mean, stddev);
        }
    }
}

void mat_fill(mat m, float val)
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

void mat_180(mat destination, mat m)
{
    assert(destination.rows == m.rows);
    assert(destination.cols == m.cols);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            mat_at(destination, i, j) = mat_at(m, m.rows - 1 - i, m.cols - 1 - j);
        }
    }
}

void mat_scale(mat destination, mat m, float a)
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

    for (int i = 0; i < m.cols; ++i) {
        float max = -FLT_MAX;
        for (int j = 0; j < m.rows; ++j) {
            if (mat_at(m, j, i) > max) max = mat_at(m, j, i);
        }

        float sum = 0;
        for (int j = 0; j < destination.rows; ++j) {
            float val = exp(mat_at(m, j, i) - max);
            mat_at(destination, j, i) = val;
            sum += val;
        }

        for (int j = 0; j < destination.rows; ++j) {
            mat_at(destination, j, i) /= sum;
        }
    }
}

void mat_pad(mat destination, mat m, padding_t padding)
{
    assert(destination.rows == m.rows + padding[TOP] + padding[BOTTOM]);
    assert(destination.cols == m.cols + padding[LEFT] + padding[RIGHT]);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            mat_at(destination, i, j) = 0;
            if (i >= padding[TOP] && i < (m.rows + padding[TOP]) &&
                j >= padding[LEFT] && j < (m.rows + padding[LEFT])) {
                mat_at(destination, i, j) = mat_at(m, i - padding[TOP], j - padding[LEFT]);
            }
        }
    }
}

void mat_convolve(mat destination, mat m, mat filter)
{
    assert(destination.rows == m.rows - filter.rows + 1);
    assert(destination.cols == m.cols - filter.cols + 1);

    mat_fill(destination, 0);

    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            for (int k = 0; k < filter.rows; ++k) {
                for (int l = 0; l < filter.cols; ++l) {
                    mat_at(destination, i, j) += mat_at(m, i + k, i + l) * mat_at(filter, k, l);
                }
            }
        }
    }
}

void mat_unflatten(tens4D destination, mat m)
{
    assert(destination.rows * destination.cols * destination.depth == m.rows);
    assert(destination.batches == m.cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < destination.batches; ++i) {
        for (int j = 0; j < destination.depth; ++j) {
            for (int k = 0; k < destination.rows; ++k) {
                for (int l = 0; l < destination.cols; ++l) {
                    int index = (j * destination.rows + k) * destination.cols + l;
                    tens4D_at(destination, k, l, j, i) = mat_at(m, index, i);
                }
            }
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

void mat_save(mat m, FILE *f)
{
    assert(f != NULL);

    fwrite(m.vals, sizeof(float), m.rows * m.cols, f);
}

void mat_load(mat m, FILE *f)
{
    assert(f != NULL);

    fread(m.vals, sizeof(float), m.rows * m.cols, f);
}
