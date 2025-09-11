#include <assert.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

tens tens2D_alloc(int rows, int cols)
{
	tens t;

	t.rows = rows;
	t.cols = cols;
    t.depth = 1;
    t.batches = 1;

	t.vals = malloc(rows * cols * sizeof(float));

	return t;
}

tens tens3D_alloc(int rows, int cols, int depth)
{
    tens t;

    t.rows = rows;
    t.cols = cols;
    t.depth = depth;
    t.batches = 1;

    t.vals = malloc(rows * cols * depth * sizeof(float));

    return t;
}

tens tens4D_alloc(int rows, int cols, int depth, int batches)
{
    tens t;

    t.rows = rows;
    t.cols = cols;
    t.depth = depth;
    t.batches = batches;

    t.vals = malloc(rows * cols * depth * batches * sizeof(float));

    return t;
}

void tens_reshape(tens dest, tens t)
{
    int elements = t.batches * t.depth * t.rows * t.cols;
    assert(dest.batches * dest.depth * dest.rows * dest.cols == elements);

    memcpy(dest.vals, t.vals, elements * sizeof(float));
}

void tens_rand(tens t, float min, float max)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.batches; ++i) {
        for (int j = 0; j < t.depth; ++j) {
            for (int k = 0; k < t.rows; ++k) {
                for (int l = 0; l < t.cols; ++l) {
                    tens4D_at(t, k, l, j, i) = rand_float(min, max);
                }
            }
        }
    }
}

void tens_normal(tens t, float mean, float stddev)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.batches; ++i) {
        for (int j = 0; j < t.depth; ++j) {
            for (int k = 0; k < t.rows; ++k) {
                for (int l = 0; l < t.cols; ++l) {
                    tens4D_at(t, k, l, j, i) = rand_normal(mean, stddev);
                }
            }
        }
    }
}

void tens_fill(tens t, float val)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.batches; ++i) {
        for (int j = 0; j < t.depth; ++j) {
            for (int k = 0; k < t.rows; ++k) {
                for (int l = 0; l < t.cols; ++l) {
                    tens4D_at(t, k, l, j, i) = val;
                }
            }
        }
    }
}

void tens_copy(tens dest, tens t)
{
	assert(dest.rows == t.rows);
	assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t, k, l, j, i);
                }
            }
		}
	}
}

void tens_add(tens dest, tens t1, tens t2)
{
	assert(t1.rows == t2.rows);
	assert(t1.cols == t2.cols);
    assert(t1.depth == t2.depth);
    assert(t1.batches == t2.batches);
	assert(dest.rows == t1.rows);
	assert(dest.cols == t1.cols);
    assert(dest.depth == t1.depth);
    assert(dest.batches == t1.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t1, k, l, j, i) + tens4D_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_sub(tens dest, tens t1, tens t2)
{
	assert(t1.rows == t2.rows);
	assert(t1.cols == t2.cols);
    assert(t1.depth == t2.depth);
    assert(t1.batches == t2.batches);
	assert(dest.rows == t1.rows);
	assert(dest.cols == t1.cols);
    assert(dest.depth == t1.depth);
    assert(dest.batches == t1.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t1, k, l, j, i) - tens4D_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_dot(tens dest, tens t1, tens t2)
{
	assert(t1.cols == t2.rows);
    assert(t1.depth == t2.depth);
    assert(t1.batches == t2.batches);
	assert(dest.rows == t1.rows);
	assert(dest.cols == t2.cols);
    assert(dest.depth == t1.depth);
    assert(dest.batches == t1.batches);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dest.batches; ++i) {
        for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = 0.0f;

                    for (int m = 0; m < t1.cols; ++m) {
                        tens4D_at(dest, k, l, j, i) += tens4D_at(t1, k, m, j, i) * tens4D_at(t2, m, l, j, i);
                    }
                }
            }
        }
    }
}

void tens_had(tens dest, tens t1, tens t2)
{
	assert(t1.rows == t2.rows);
	assert(t1.cols == t2.cols);
    assert(t1.depth == t2.depth);
    assert(t1.batches == t2.batches);
	assert(dest.rows == t1.rows);
	assert(dest.cols == t1.cols);
    assert(dest.depth == t1.depth);
    assert(dest.batches == t1.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t1, k, l, j, i) * tens4D_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_trans(tens dest, tens t)
{
	assert(dest.rows == t.rows);
	assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t, l, k, j, i);
                }
            }
		}
	}
}

void tens_180(tens dest, tens t)
{
	assert(dest.rows == t.rows);
	assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = tens4D_at(t, t.rows - 1 - k, t.cols - 1 - l, j, i);
                }
            }
		}
	}
}

void tens_scale(tens dest, tens t, float a)
{
	assert(dest.rows == t.rows);
	assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = a * tens4D_at(t, k, l, j, i);
                }
            }
		}
	}
}

void tens_func(tens dest, tens t, func f)
{
	assert(dest.rows == t.rows);
	assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.batches; ++i) {
		for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = f(tens4D_at(t, k, l, j, i));
                }
            }
		}
	}
}

void tens_pad(tens dest, tens t)
{
    assert(dest.rows >= t.rows);
    assert(dest.cols >= t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    int top_padding = (dest.rows - t.rows) / 2;
    int left_padding = (dest.cols - t.cols) / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dest.batches; ++i) {
        for (int j = 0; j < dest.depth; ++j) {
            for (int k = 0; k < dest.rows; ++k) {
                for (int l = 0; l < dest.cols; ++l) {
                    tens4D_at(dest, k, l, j, i) = 0.0f;

                    if (k >= top_padding && k < (t.rows + top_padding) &&
                        l >= left_padding && l < (t.cols + left_padding)) {
                            tens4D_at(dest, k, l, j, i) = tens4D_at(t, k - top_padding, l - left_padding, j, i);
                    }
                }
            }
        }
    }
}

void tens_softmax(tens dest, tens t)
{
    assert(dest.rows == t.rows);
    assert(dest.cols == t.cols);
    assert(dest.depth == t.depth);
    assert(dest.batches == t.batches);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.batches; ++i) {
        for (int j = 0; j < t.depth; ++j) {
            for (int k = 0; k < t.cols; ++k) {
                float max = -FLT_MAX;

                for (int l = 0; l < t.rows; ++l) {
                    if (tens4D_at(t, l, k, j, i) > max) max = tens4D_at(t, l, k, j, i);
                }

                float sum = 0.0f;

                for (int l = 0; l < t.rows; ++l) {
                    float val = expf(tens4D_at(t, l, k, j, i) - max);

                    tens4D_at(dest, l, k, j, i) = val;

                    sum += val;
                }

                for (int l = 0; l < t.rows; ++l) {
                    tens4D_at(dest, l, k, j, i) /= sum;
                }
            }
        }
    }
}

void tens_destroy(tens t)
{
    free(t.vals);
}

void tens_print(tens t)
{
    for (int i = 0; i < t.batches; ++i) {
        for (int j = 0; j < t.depth; ++j) {
            for (int k = 0; k < t.rows; ++k) {
                for (int l = 0; l < t.cols; ++l) {
                    printf("%f ", tens4D_at(t, k, l, j, i));
                }
                printf("\n");
            }
        }
    }
}

void tens_save(tens t, FILE *f)
{
    assert(f != NULL);

    fwrite(t.vals, sizeof(float), t.batches * t.depth * t.rows * t.cols, f);
}

void tens_load(tens t, FILE *f)
{
    assert(f != NULL);

    fread(t.vals, sizeof(float), t.batches * t.depth * t.rows * t.cols, f);
}
