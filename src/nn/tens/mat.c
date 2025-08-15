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

void mat_copy(mat dest, mat m)
{
	assert(dest.rows == m.rows);
	assert(dest.cols == m.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = mat_at(m, i, j);
		}
	}
}

void mat_add(mat dest, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(dest.rows == m1.rows);
	assert(dest.cols == m1.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = mat_at(m1, i, j) + mat_at(m2, i, j);
		}
	}
}

void mat_sub(mat dest, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(dest.rows == m1.rows);
	assert(dest.cols == m1.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = mat_at(m1, i, j) - mat_at(m2, i, j);
		}
	}
}

void mat_dot(mat dest, mat m1, mat m2)
{
	assert(m1.cols == m2.rows);
	assert(dest.rows == m1.rows);
	assert(dest.cols == m2.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = 0;
			for (int k = 0; k < m1.cols; ++k) {
				mat_at(dest, i, j) += mat_at(m1, i, k) * mat_at(m2, k, j);
			}
		}
	}
}

void mat_had(mat dest, mat m1, mat m2)
{
	assert(m1.rows == m2.rows);
	assert(m1.cols == m2.cols);
	assert(dest.rows == m1.rows);
	assert(dest.cols == m1.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = mat_at(m1, i, j) * mat_at(m2, i, j);
		}
	}
}

void mat_trans(mat dest, mat m)
{
	assert(dest.rows == m.cols);
	assert(dest.cols == m.rows);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = mat_at(m, j, i);
		}
	}
}

void mat_180(mat dest, mat m)
{
    assert(dest.rows == m.rows);
    assert(dest.cols == m.cols);


    for (int i = 0; i < dest.rows; ++i) {
        for (int j = 0; j < dest.cols; ++j) {
            mat_at(dest, i, j) = mat_at(m, m.rows - 1 - i, m.cols - 1 - j);
        }
    }
}

void mat_scale(mat dest, mat m, float a)
{
	assert(dest.rows == m.rows);
	assert(dest.cols == m.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = a * mat_at(m, i, j);
		}
	}
}

void mat_func(mat dest, mat m, func f)
{
	assert(dest.rows == m.rows);
	assert(dest.cols == m.cols);


	for (int i = 0; i < dest.rows; ++i) {
		for (int j = 0; j < dest.cols; ++j) {
			mat_at(dest, i, j) = f(mat_at(m, i, j));
		}
	}
}

void mat_pad(mat dest, mat m, padding_t padding)
{
    assert(dest.rows == m.rows + padding[TOP] + padding[BOTTOM]);
    assert(dest.cols == m.cols + padding[LEFT] + padding[RIGHT]);


    for (int i = 0; i < dest.rows; ++i) {
        for (int j = 0; j < dest.cols; ++j) {
            mat_at(dest, i, j) = 0;
            if (i >= padding[TOP] && i < (m.rows + padding[TOP]) &&
                j >= padding[LEFT] && j < (m.cols + padding[LEFT])) {
                mat_at(dest, i, j) = mat_at(m, i - padding[TOP], j - padding[LEFT]);
            }
        }
    }
}

void mat_convolve(mat dest, mat m, mat filter)
{
    assert(dest.rows == m.rows - filter.rows + 1);
    assert(dest.cols == m.cols - filter.cols + 1);

    mat_fill(dest, 0);

    for (int i = 0; i < dest.rows; ++i) {
        for (int j = 0; j < dest.cols; ++j) {
            for (int k = 0; k < filter.rows; ++k) {
                for (int l = 0; l < filter.cols; ++l) {
                    mat_at(dest, i, j) += mat_at(m, i + k, j + l) * mat_at(filter, k, l);
                }
            }
        }
    }
}

void mat_softmax(mat dest, mat m)
{
    assert(dest.rows == m.rows);
    assert(dest.cols == m.cols);

    for (int i = 0; i < m.cols; ++i) {
        float max = -FLT_MAX;

        for (int j = 0; j < m.rows; ++j) {
            if (mat_at(m, j, i) > max) max = mat_at(m, j, i);
        }

        float sum = 0.0f;

        for (int j = 0; j < m.rows; ++j) {
            float val = expf(mat_at(m, j, i) - max);

            mat_at(dest, j, i) = val;

            sum += val;
        }

        for (int j = 0; j < m.rows; ++j) {
            mat_at(dest, j, i) /= sum;
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
