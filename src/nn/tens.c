#include <assert.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

tens tens_alloc(int r, int c, int d, int b)
{
    tens t;

    t.dims[R] = r;
    t.dims[C] = c;
    t.dims[D] = d;
    t.dims[B] = b;

    t.vals = malloc(r * c * d * b * sizeof(float));

    return t;
}

void tens_reshape(tens dest, tens t)
{
    int elements = t.dims[B] * t.dims[D] * t.dims[R] * t.dims[C];
    assert(dest.dims[B] * dest.dims[D] * dest.dims[R] * dest.dims[C] == elements);

    memcpy(dest.vals, t.vals, elements * sizeof(float));
}

void tens_rand(tens t, float min, float max)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.dims[B]; ++i) {
        for (int j = 0; j < t.dims[D]; ++j) {
            for (int k = 0; k < t.dims[R]; ++k) {
                for (int l = 0; l < t.dims[C]; ++l) {
                    tens_at(t, k, l, j, i) = rand_float(min, max);
                }
            }
        }
    }
}

void tens_normal(tens t, float mean, float stddev)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.dims[B]; ++i) {
        for (int j = 0; j < t.dims[D]; ++j) {
            for (int k = 0; k < t.dims[R]; ++k) {
                for (int l = 0; l < t.dims[C]; ++l) {
                    tens_at(t, k, l, j, i) = rand_normal(mean, stddev);
                }
            }
        }
    }
}

void tens_fill(tens t, float val)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.dims[B]; ++i) {
        for (int j = 0; j < t.dims[D]; ++j) {
            for (int k = 0; k < t.dims[R]; ++k) {
                for (int l = 0; l < t.dims[C]; ++l) {
                    tens_at(t, k, l, j, i) = val;
                }
            }
        }
    }
}

void tens_copy(tens dest, tens t)
{
	assert(dest.dims[R] == t.dims[R]);
	assert(dest.dims[C] == t.dims[C]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = tens_at(t, k, l, j, i);
                }
            }
		}
	}
}

void tens_add(tens dest, tens t1, tens t2)
{
	assert(t1.dims[R] == t2.dims[R]);
	assert(t1.dims[C] == t2.dims[C]);
    assert(t1.dims[D] == t2.dims[D]);
    assert(t1.dims[B] == t2.dims[B]);
	assert(dest.dims[R] == t1.dims[R]);
	assert(dest.dims[C] == t1.dims[C]);
    assert(dest.dims[D] == t1.dims[D]);
    assert(dest.dims[B] == t1.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = tens_at(t1, k, l, j, i) + tens_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_sub(tens dest, tens t1, tens t2)
{
	assert(t1.dims[R] == t2.dims[R]);
	assert(t1.dims[C] == t2.dims[C]);
    assert(t1.dims[D] == t2.dims[D]);
    assert(t1.dims[B] == t2.dims[B]);
	assert(dest.dims[R] == t1.dims[R]);
	assert(dest.dims[C] == t1.dims[C]);
    assert(dest.dims[D] == t1.dims[D]);
    assert(dest.dims[B] == t1.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = tens_at(t1, k, l, j, i) - tens_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_dot(tens dest, tens t1, tens t2)
{
	assert(t1.dims[C] == t2.dims[R]);
    assert(t1.dims[D] == t2.dims[D]);
    assert(t1.dims[B] == t2.dims[B]);
	assert(dest.dims[R] == t1.dims[R]);
	assert(dest.dims[C] == t2.dims[C]);
    assert(dest.dims[D] == t1.dims[D]);
    assert(dest.dims[B] == t1.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dest.dims[B]; ++i) {
        for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < t1.dims[C]; ++m) {
                        sum += tens_at(t1, k, m, j, i) * tens_at(t2, m, l, j, i);
                    }

                    tens_at(dest, k, l, j, i) = sum;
                }
            }
        }
    }
}

void tens_had(tens dest, tens t1, tens t2)
{
	assert(t1.dims[R] == t2.dims[R]);
	assert(t1.dims[C] == t2.dims[C]);
    assert(t1.dims[D] == t2.dims[D]);
    assert(t1.dims[B] == t2.dims[B]);
	assert(dest.dims[R] == t1.dims[R]);
	assert(dest.dims[C] == t1.dims[C]);
    assert(dest.dims[D] == t1.dims[D]);
    assert(dest.dims[B] == t1.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = tens_at(t1, k, l, j, i) * tens_at(t2, k, l, j, i);
                }
            }
		}
	}
}

void tens_trans(tens dest, tens t, int perm[4])
{
	assert(dest.dims[R] == t.dims[perm[0]]);
	assert(dest.dims[C] == t.dims[perm[1]]);
    assert(dest.dims[D] == t.dims[perm[2]]);
    assert(dest.dims[B] == t.dims[perm[3]]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    int dest_index[4] = { k, l, j, i };
                    int t_index[4];

                    for (int m = 0; m < 4; ++m) {
                        t_index[perm[m]] = dest_index[m];
                    }

                    tens_at(dest, k, l, j, i) = tens_at(t, t_index[0], t_index[1], t_index[2], t_index[3]);
                }
            }
		}
	}
}

void tens_180(tens dest, tens t, int flip[4])
{
	assert(dest.dims[R] == t.dims[R]);
	assert(dest.dims[C] == t.dims[C]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {

                    int t_index[4] = { k, l, j, i };

                    for (int m = 0; m < 4; ++m) {
                        if (flip[m]) {
                            t_index[m] = t.dims[m] - 1 - t_index[m];
                        }
                    }

                    tens_at(dest, k, l, j, i) = tens_at(t, t_index[0], t_index[1], t_index[2], t_index[3]);
                }
            }
		}
	}
}

void tens_scale(tens dest, tens t, float a)
{
	assert(dest.dims[R] == t.dims[R]);
	assert(dest.dims[C] == t.dims[C]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = a * tens_at(t, k, l, j, i);
                }
            }
		}
	}
}

void tens_func(tens dest, tens t, func f)
{
	assert(dest.dims[R] == t.dims[R]);
	assert(dest.dims[C] == t.dims[C]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < dest.dims[B]; ++i) {
		for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = f(tens_at(t, k, l, j, i));
                }
            }
		}
	}
}

void tens_pad(tens dest, tens t, int padding[4])
{
    assert(dest.dims[R] == t.dims[R] + padding[TOP] + padding[BOTTOM]);
    assert(dest.dims[C] == t.dims[C] + padding[LEFT] + padding[RIGHT]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dest.dims[B]; ++i) {
        for (int j = 0; j < dest.dims[D]; ++j) {
            for (int k = 0; k < dest.dims[R]; ++k) {
                for (int l = 0; l < dest.dims[C]; ++l) {
                    tens_at(dest, k, l, j, i) = 0.0f;

                    if (k >= padding[TOP] && k < (t.dims[R] + padding[TOP]) &&
                        l >= padding[LEFT] && l < (t.dims[C] + padding[LEFT])) {
                            tens_at(dest, k, l, j, i) = tens_at(t, k - padding[TOP], l - padding[LEFT], j, i);
                    }
                }
            }
        }
    }
}

void tens_softmax(tens dest, tens t)
{
    assert(dest.dims[R] == t.dims[R]);
    assert(dest.dims[C] == t.dims[C]);
    assert(dest.dims[D] == t.dims[D]);
    assert(dest.dims[B] == t.dims[B]);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < t.dims[B]; ++i) {
        for (int j = 0; j < t.dims[D]; ++j) {
            for (int k = 0; k < t.dims[C]; ++k) {
                float max = -FLT_MAX;

                for (int l = 0; l < t.dims[R]; ++l) {
                    if (tens_at(t, l, k, j, i) > max) max = tens_at(t, l, k, j, i);
                }

                float sum = 0.0f;

                for (int l = 0; l < t.dims[R]; ++l) {
                    float val = expf(tens_at(t, l, k, j, i) - max);

                    tens_at(dest, l, k, j, i) = val;

                    sum += val;
                }

                for (int l = 0; l < t.dims[R]; ++l) {
                    tens_at(dest, l, k, j, i) /= sum;
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
    for (int i = 0; i < t.dims[B]; ++i) {
        for (int j = 0; j < t.dims[D]; ++j) {
            for (int k = 0; k < t.dims[R]; ++k) {
                for (int l = 0; l < t.dims[C]; ++l) {
                    printf("%f ", tens_at(t, k, l, j, i));
                }
                printf("\n");
            }
        }
    }
}

void tens_save(tens t, FILE *f)
{
    assert(f != NULL);

    fwrite(t.vals, sizeof(float), t.dims[B] * t.dims[D] * t.dims[R] * t.dims[C], f);
}

void tens_load(tens t, FILE *f)
{
    assert(f != NULL);

    fread(t.vals, sizeof(float), t.dims[B] * t.dims[D] * t.dims[R] * t.dims[C], f);
}
