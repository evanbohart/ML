#include "nn.h"
#include <math.h>
#include <assert.h>

cnet cnet_alloc(int layers, mat convolutions, mat input_dims, int filter_size)
{
    assert(convolutions.rows == layers - 1);
    assert(convolutions.cols == 1);
    assert(input_dims.rows == 3);
    assert(input_dims.cols == 1);

    cnet cn;

    cn.layers = layers;

    cn.convolutions = mat_alloc(convolutions.rows, convolutions.cols);
    mat_copy(cn.convolutions, convolutions);

    cn.input_dims = mat_alloc(input_dims.rows, input_dims.cols);
    mat_copy(cn.input_dims, input_dims);

    cn.lins = malloc((layers - 1) * sizeof(tens));
    assert(cn.lins);

    cn.lins[0] = tens_alloc(mat_at(input_dims, 0, 0) - filter_size + 1,
                            mat_at(input_dims, 1, 0) - filter_size + 1,
                            mat_at(convolutions, 0, 0));
    for (int i = 1; i < layers - 1; ++i) {
        cn.lins[i] = tens_alloc(cn.lins[i - 1].rows - filter_size + 1,
                                cn.lins[i - 1].cols - filter_size + 1,
                                mat_at(convolutions, i, 0));
    }

    cn.acts = malloc((layers - 1) * sizeof(tens));
    assert(cn.acts);

    for (int i = 0; i < layers - 1; ++i) {
        cn.acts[i] = tens_alloc(cn.lins[i].rows, cn.lins[i].cols, cn.lins[i].depth);
    }

    cn.filters = malloc((layers - 1) * sizeof(tens *));
    assert(cn.filters);

    cn.filters[0] = malloc(mat_at(convolutions, 0, 0) * sizeof(tens));
    assert(cn.filters[0]);

    for (int i = 0; i < mat_at(convolutions, 0, 0); ++i) {
        cn.filters[0][i] = tens_alloc(filter_size, filter_size, mat_at(input_dims, 2, 0));
    }

    for (int i = 1; i < layers - 1; ++i) {
        cn.filters[i] = malloc(mat_at(convolutions, i, 0) * sizeof(tens));
        assert(cn.filters[i]);

        for (int j = 0; j < mat_at(convolutions, i, 0); ++j) {
            cn.filters[i][j] = tens_alloc(filter_size, filter_size, mat_at(convolutions, i - 1, 0));
        }
    }

    cn.biases = malloc((layers - 1) * sizeof(tens));
    assert(cn.biases);

    for (int i = 0; i < layers - 1; ++i) {
        cn.biases[i] = tens_alloc(cn.lins[i].rows, cn.lins[i].cols, cn.lins[i].depth);
    }

    cn.filter_size = filter_size;

    cn.actfuncs = malloc((layers - 1) * sizeof(actfunc));
    assert(cn.actfuncs);

    return cn;
}

void cnet_destroy(cnet *cn)
{
    for (int i = 0; i < cn->layers - 1; ++i) {
        tens_destroy(&cn->lins[i]);
        tens_destroy(&cn->acts[i]);

        for (int j = 0; j < mat_at(cn->convolutions, i, 0); ++j) {
            tens_destroy(&cn->filters[i][j]);
        }

        free(cn->filters[i]);

        tens_destroy(&cn->biases[i]);
    }

    free(cn->lins);
    free(cn->acts);
    free(cn->filters);
    free(cn->biases);
    free(cn->actfuncs);
}

void cnet_copy(cnet destination, cnet cn)
{
    assert(destination.layers == cn.layers);
    assert(mat_compare(destination.convolutions, cn.convolutions));
    assert(mat_compare(destination.input_dims, cn.input_dims));
    assert(destination.filter_size == cn.filter_size);

    for (int i = 0; i < destination.layers - 1; ++i) {
        tens_copy(destination.lins[i], cn.lins[i]);
        tens_copy(destination.acts[i], cn.acts[i]);

        for (int j = 0; j < mat_at(destination.convolutions, i, 0); ++j) {
            tens_copy(destination.filters[i][j], cn.filters[i][j]);
        }

        tens_copy(destination.biases[i], cn.biases[i]);
    }
}

void cnet_print(cnet cn)
{
    for (int i = 0; i < cn.layers - 1; ++i) {
        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            tens_print(cn.filters[i][j]);
        }

        tens_print(cn.biases[i]);
    }

}

void cnet_load(cnet *cn, FILE *f)
{
    for (int i = 0; i < cn->layers - 1; ++i) {
        tens_load(&cn->lins[i], f);
        tens_load(&cn->acts[i], f);

        for (int j = 0; j < mat_at(cn->convolutions, i, 0); ++j) {
            tens_load(&cn->filters[i][j], f);
        }

        tens_load(&cn->biases[i], f);
    }
}

void cnet_save(cnet cn, FILE *f)
{
    for (int i = 0; i < cn.layers - 1; ++i) {
        tens_save(cn.lins[i], f);
        tens_save(cn.acts[i], f);

        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            tens_save(cn.filters[i][j], f);
        }

        tens_save(cn.biases[i], f);
    }
}

void cnet_glorot(cnet cn)
{
    for (int i = 0; i < mat_at(cn.convolutions, 0, 0); ++i) {
        tens_normal(cn.filters[0][i], 0, 2 / (mat_at(cn.input_dims, 2, 0) +
                                              mat_at(cn.convolutions, 0, 0)));
    }

    for (int i = 1; i < cn.layers - 1; ++i) {
        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            tens_normal(cn.filters[i][j], 0, 2 / (mat_at(cn.convolutions, i - 1, 0) +
                                                  mat_at(cn.convolutions, i, 0)));
        }
    }

    for (int i = 0; i < cn.layers - 1; ++i) {
        tens_fill(cn.biases[i], 0);
    }
}

void cnet_he(cnet cn)
{
    for (int i = 0; i < mat_at(cn.convolutions, 0, 0); ++i) {
        tens_normal(cn.filters[0][i], 0, 2 / mat_at(cn.input_dims, 2, 0));
    }

    for (int i = 1; i < cn.layers - 1; ++i) {
        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            tens_normal(cn.filters[i][j], 0, 2 / mat_at(cn.convolutions, i - 1, 0));
        }
    }

    for (int i = 0; i < cn.layers - 1; ++i) {
        tens_fill(cn.biases[i], 0);
    }
}

void cnet_forward(cnet cn, tens inputs)
{
    assert(inputs.rows == mat_at(cn.input_dims, 0, 0));
    assert(inputs.cols == mat_at(cn.input_dims, 1, 0));
    assert(inputs.depth == mat_at(cn.input_dims, 2, 0));

    mat convolved = mat_alloc(cn.lins[0].rows, cn.lins[0].cols);

    tens_fill(cn.lins[0], 0);
    for (int i = 0; i < mat_at(cn.convolutions, 0, 0); ++i) {
        for (int j = 0; j < cn.filters[0][i].depth; ++j) {
            mat_convolve(convolved, inputs.mats[j], cn.filters[0][i].mats[j]);
            mat_add(cn.lins[0].mats[i], cn.lins[0].mats[i], convolved);
        }
    }

    tens_add(cn.lins[0], cn.lins[0], cn.biases[0]);

    switch (cn.actfuncs[0]) {
        case SIGMOID:
            tens_func(cn.acts[0], cn.lins[0], sig);
            break;
        case RELU:
            tens_func(cn.acts[0], cn.lins[0], relu);
            break;
        case SOFTMAX:
            break;
    }

    free(convolved.vals);

    for (int i = 1; i < cn.layers - 1; ++i) {
        convolved = mat_alloc(cn.lins[i].rows, cn.lins[i].cols);

        tens_fill(cn.lins[i], 0);
        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            for (int k = 0; k < cn.filters[i][j].depth; ++k) {
                mat_convolve(convolved, cn.acts[i - 1].mats[k], cn.filters[i][j].mats[k]);
                mat_add(cn.lins[i].mats[j], cn.lins[i].mats[j], convolved);
            }
        }

        tens_add(cn.lins[i], cn.lins[i], cn.biases[i]);

        switch (cn.actfuncs[i]) {
            case SIGMOID:
                tens_func(cn.acts[i], cn.lins[i], sig);
                break;
            case RELU:
                tens_func(cn.acts[i], cn.lins[i], relu);
                break;
            case SOFTMAX:
                break;
        }

        free(convolved.vals);
    }
}

void cnet_backprop(cnet cn, tens inputs, mat delta, double rate)
{
    assert(inputs.rows == mat_at(cn.input_dims, 0, 0));
    assert(inputs.cols == mat_at(cn.input_dims, 1, 0));
    assert(inputs.depth == mat_at(cn.input_dims, 2, 0));
    assert(delta.rows == cn.acts[cn.layers - 2].depth);
    assert(delta.cols == 1);

    tens *deltas = malloc((cn.layers - 1) * sizeof(tens));
    assert(deltas);

    deltas[cn.layers - 2] = tens_alloc(cn.acts[cn.layers - 2].rows,
                                       cn.acts[cn.layers - 2].cols,
                                       cn.acts[cn.layers - 2].depth);
    tens_fill(deltas[cn.layers - 2], 0);

    cnet_forward(cn, inputs);

    for (int i = 0; i < cn.acts[cn.layers - 2].depth; ++i) {
        int row = 0;
        int col = 0;
        int target = mat_max(cn.acts[cn.layers - 2].mats[i]);

        for (int j = 0; j < cn.acts[cn.layers - 2].rows; ++j) {
            for (int k = 0; k < cn.acts[cn.layers - 2].cols; ++k) {
                if (tens_at(cn.acts[cn.layers - 2], j, k, i) == target) {
                    row = j;
                    col = k;
                }
            }
        }

        tens_at(deltas[cn.layers - 2], row, col, i) = mat_at(delta, i, 0);

        switch (cn.actfuncs[cn.layers - 2]) {
            case SIGMOID:
                tens_at(deltas[cn.layers - 2], row, col, i) *=
                       dsig(tens_at(cn.lins[cn.layers - 2], row, col, i));
                break;
            case RELU:
                tens_at(deltas[cn.layers - 2], row, col, i) *=
                        drelu(tens_at(cn.lins[cn.layers - 2], row, col, i));
                break;
            case SOFTMAX:
                break;
        }
    }

    for (int i = cn.layers - 2; i >= 1; --i) {
        tens padded = tens_alloc(deltas[i].rows + 2 * (cn.filter_size - 1),
                                 deltas[i].cols + 2 * (cn.filter_size - 1),
                                 deltas[i].depth);
        tens_pad(padded, deltas[i]);

        mat filter_trans = mat_alloc(cn.filter_size, cn.filter_size);
        mat convolved = mat_alloc(deltas[i].rows + filter_trans.rows - 1,
                                  deltas[i].cols + filter_trans.cols - 1);

        deltas[i - 1] = tens_alloc(cn.acts[i - 1].rows, cn.acts[i - 1].cols, cn.acts[i - 1].depth);
        tens_fill(deltas[i - 1], 0);

        for (int j = 0; j < deltas[i - 1].depth; ++j) {
            for (int k = 0; k < mat_at(cn.convolutions, i, 0); ++k) {
                mat_trans(filter_trans, cn.filters[i][k].mats[j]);
                mat_convolve(convolved, padded.mats[k], filter_trans);

                mat_add(deltas[i - 1].mats[j], deltas[i - 1].mats[j], convolved);
            }
        }

        tens_destroy(&padded);
        free(filter_trans.vals);
        free(convolved.vals);
    }

    tens dfilter = tens_alloc(cn.filter_size, cn.filter_size, cn.filters[0][0].depth);
    tens dbias = tens_alloc(cn.biases[0].rows, cn.biases[0].cols, cn.biases[0].depth);

    for (int i = 0; i < mat_at(cn.convolutions, 0, 0); ++i) {
        for (int j = 0; j < cn.filters[0][i].depth; ++j) {
            mat_convolve(dfilter.mats[j], inputs.mats[j], deltas[0].mats[i]);
        }

        tens_func(dfilter, dfilter, clip);
        tens_scale(dfilter, dfilter, rate);
        tens_sub(cn.filters[0][i], cn.filters[0][i], dfilter);
    }

    tens_copy(dbias, deltas[0]);
    tens_func(dbias, dbias, clip);
    tens_scale(dbias, dbias, rate);
    tens_sub(cn.biases[0], cn.biases[0], dbias);

    tens_destroy(&dfilter);
    tens_destroy(&dbias);
    tens_destroy(&deltas[0]);

    for (int i = 1; i < cn.layers - 1; ++i) {
        dfilter = tens_alloc(cn.filter_size, cn.filter_size, cn.filters[i][0].depth);
        dbias = tens_alloc(cn.biases[i].rows, cn.biases[i].cols, cn.biases[i].depth);

        for (int j = 0; j < mat_at(cn.convolutions, i, 0); ++j) {
            for (int k = 0; k < cn.filters[i][j].depth; ++k) {
                mat_convolve(dfilter.mats[k], cn.acts[i - 1].mats[k], deltas[i].mats[j]); 
            }

            tens_func(dfilter, dfilter, clip);
            tens_scale(dfilter, dfilter, rate);
            tens_sub(cn.filters[i][j], cn.filters[i][j], dfilter);
        }

        tens_copy(dbias, deltas[i]);
        tens_func(dbias, dbias, clip);
        tens_scale(dbias, dbias, rate);
        tens_sub(cn.biases[i], cn.biases[i], dbias);

        tens_destroy(&dfilter);
        tens_destroy(&dbias);
        tens_destroy(&deltas[i]);
    }

    free(deltas);
}
