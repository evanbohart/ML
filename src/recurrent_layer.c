#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer recurrent_layer_alloc(int input_size, int hidden_size,
                            int output_size, int batch_size, int steps,
                            actfunc activation_hidden, actfunc activation_output)
{
    recurrent_layer *rl = malloc(sizeof(recurrent_layer));
    rl->input_size = input_size;
    rl->hidden_size = hidden_size;
    rl->output_size = output_size;
    rl->batch_size = batch_size;
    rl->steps = steps;
    rl->weights_input = mat_alloc(hidden_size, input_size);
    rl->weights_hidden = mat_alloc(hidden_size, hidden_size);
    rl->weights_output = mat_alloc(output_size, hidden_size);
    rl->biases_hidden = mat_alloc(hidden_size, 1);
    rl->biases_output = mat_alloc(output_size, 1);
    rl->initial_hidden = mat_alloc(hidden_size, 1);
    rl->activation_hidden = activation_hidden;
    rl->activation_output = activation_output;

    rl->input_cache = tens3D_alloc(input_size, batch_size, steps);
    rl->lins_cache_hidden = tens3D_alloc(hidden_size, batch_size, steps);
    rl->acts_cache_hidden = tens3D_alloc(hidden_size, batch_size, steps);
    rl->lins_cache_output = tens3D_alloc(output_size, batch_size, steps);

    layer l;
    l.type = RECURRENT;
    l.data = rl;
    l.forward = recurrent_forward;
    l.backprop = recurrent_backprop;
    l.destroy = recurrent_destroy;
    l.he = recurrent_he;
    l.glorot = recurrent_glorot;
    l.print = recurrent_print;
    l.save = recurrent_save;
    l.load = recurrent_load;

    return l;
}

void recurrent_forward(layer l, void *input, void **output)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;
    tens3D *tens3D_input = (tens3D *)input;

    assert(tens3D_input->rows == rl->input_size);
    assert(tens3D_input->cols == rl->batch_size);
    assert(tens3D_input->depth == rl->steps);

    tens3D *tens3D_output = malloc(sizeof(tens3D));
    *tens3D_output = tens3D_alloc(rl->output_size, rl->batch_size, rl->steps);

    mat dot_input = mat_alloc(rl->hidden_size, rl->batch_size);
    mat dot_hidden = mat_alloc(rl->hidden_size, rl->batch_size);

    for (int i = 0; i < rl->steps; ++i) {
        mat_dot(dot_input, rl->weights_input, tens3D_input->mats[i]);

        if (i > 0) {
            mat_dot(dot_hidden, rl->weights_hidden, rl->acts_cache_hidden.mats[i - 1]);
        }
        else {
            mat broadcasted = mat_alloc(rl->hidden_size, rl->batch_size);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < rl->hidden_size; ++j) {
                for (int k = 0; k < rl->batch_size; ++k) {
                    mat_at(broadcasted, j, k) = mat_at(rl->initial_hidden, j, 0);
                }
            }

            mat_dot(dot_hidden, rl->weights_hidden, broadcasted);
 
            free(broadcasted.vals);
        }

        mat_add(rl->lins_cache_hidden.mats[i], dot_input, dot_hidden);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < rl->hidden_size; ++j) {
            for (int k = 0; k < rl->batch_size; ++k) {
                tens3D_at(rl->lins_cache_hidden, j, k, i) += mat_at(rl->biases_hidden, j, 0);
            }
        }

        switch (rl->activation_hidden) {
            case LIN:
                mat_func(rl->acts_cache_hidden.mats[i], rl->lins_cache_hidden.mats[i], lin);
                break;
            case SIG:
                mat_func(rl->acts_cache_hidden.mats[i], rl->lins_cache_hidden.mats[i], sig);
                break;
            case TANH:
                mat_func(rl->acts_cache_hidden.mats[i], rl->lins_cache_hidden.mats[i], tanhf);
                break;
            case RELU:
                mat_func(rl->acts_cache_hidden.mats[i], rl->lins_cache_hidden.mats[i], relu);
                break;
        }

        mat_dot(rl->lins_cache_output.mats[i], rl->weights_output, rl->acts_cache_hidden.mats[i]);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < rl->output_size; ++j) {
            for (int k = 0; k < rl->batch_size; ++k) {
                tens3D_at(rl->lins_cache_output, j, k, i) += mat_at(rl->biases_output, j, 0);
            }
        }

        switch (rl->activation_output) {
            case LIN:
                mat_func(tens3D_output->mats[i], rl->lins_cache_output.mats[i], lin);
                break;
            case SIG:
                mat_func(tens3D_output->mats[i], rl->lins_cache_output.mats[i], sig);
                break;
            case TANH:
                mat_func(tens3D_output->mats[i], rl->lins_cache_output.mats[i], tanhf);
                break;
            case RELU:
                mat_func(tens3D_output->mats[i], rl->lins_cache_output.mats[i], relu);
                break;
        }
    }

    *output = tens3D_output;

    free(dot_input.vals);
    free(dot_hidden.vals);
}

void recurrent_backprop(layer l, void *grad_in, void **grad_out, float rate)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;
    tens3D *tens3D_grad_in = (tens3D *)grad_in;

    assert(tens3D_grad_in->rows == rl->output_size);
    assert(tens3D_grad_in->cols == rl->batch_size);
    assert(tens3D_grad_in->depth == rl->steps);

    tens3D *tens3D_grad_out = malloc(sizeof(tens3D));
    *tens3D_grad_out = tens3D_alloc(rl->input_size, rl->batch_size, rl->steps);

    tens3D delta_hidden = tens3D_alloc(rl->hidden_size, rl->batch_size, rl->steps);
    tens3D delta_output = tens3D_alloc(rl->output_size, rl->batch_size, rl->steps);

    tens3D input_trans = tens3D_alloc(rl->batch_size, rl->input_size, rl->steps);
    tens3D_trans(input_trans, rl->input_cache);
    tens3D hidden_acts_trans = tens3D_alloc(rl->batch_size, rl->hidden_size, rl->steps);
    tens3D_trans(hidden_acts_trans, rl->acts_cache_hidden);
    mat initial_hidden_trans = mat_alloc(1, rl->hidden_size);
    mat_trans(initial_hidden_trans, rl->initial_hidden);

    tens3D lins_deriv_hidden = tens3D_alloc(rl->hidden_size, rl->batch_size, rl->steps);

    switch (rl->activation_hidden) {
        case LIN:
            tens3D_func(lins_deriv_hidden, rl->lins_cache_hidden, dlin);
            break;
        case SIG:
            tens3D_func(lins_deriv_hidden, rl->lins_cache_hidden, dsig);
            break;
        case TANH:
            tens3D_func(lins_deriv_hidden, rl->lins_cache_hidden, dtanh);
            break;
        case RELU:
            tens3D_func(lins_deriv_hidden, rl->lins_cache_hidden, drelu);
            break;
    }

    tens3D lins_deriv_output = tens3D_alloc(rl->output_size, rl->batch_size, rl->steps);

    switch (rl->activation_output) {
        case LIN:
            tens3D_func(lins_deriv_output, rl->lins_cache_hidden, dlin);
            break;
        case SIG:
            tens3D_func(lins_deriv_output, rl->lins_cache_hidden, dsig);
            break;
        case TANH:
            tens3D_func(lins_deriv_output, rl->lins_cache_hidden, dtanh);
            break;
        case RELU:
            tens3D_func(lins_deriv_output, rl->lins_cache_hidden, drelu);
            break;
    }

    mat weights_input_trans = mat_alloc(rl->input_size, rl->hidden_size);
    mat_trans(weights_input_trans, rl->weights_input);
    mat weights_hidden_trans = mat_alloc(rl->hidden_size, rl->hidden_size);
    mat_trans(weights_hidden_trans, rl->weights_hidden);
    mat weights_output_trans = mat_alloc(rl->hidden_size, rl->output_size);
    mat_trans(weights_output_trans, rl->weights_output);

    mat temp = mat_alloc(rl->hidden_size, rl->batch_size);

    mat dw_input_current = mat_alloc(rl->hidden_size, rl->input_size);
    mat dw_hidden_current = mat_alloc(rl->hidden_size, rl->hidden_size);
    mat dw_output_current = mat_alloc(rl->output_size, rl->hidden_size);
    mat dw_input = mat_alloc(rl->hidden_size, rl->input_size);
    mat_fill(dw_input, 0);
    mat dw_hidden = mat_alloc(rl->hidden_size, rl->hidden_size);
    mat_fill(dw_hidden, 0);
    mat dw_output = mat_alloc(rl->output_size, rl->hidden_size);
    mat_fill(dw_output, 0);

    mat db_hidden = mat_alloc(rl->hidden_size, 1);
    mat_fill(db_hidden, 0);
    mat db_output = mat_alloc(rl->output_size, 1);
    mat_fill(db_output, 0);

    mat dinitial_hidden = mat_alloc(rl->hidden_size, 1);
    mat_fill(dinitial_hidden, 0);

    for (int i = rl->steps - 1; i >= 0; --i) {
        mat_had(delta_output.mats[i], tens3D_grad_in->mats[i], lins_deriv_output.mats[i]);
        mat_dot(dw_output_current, delta_output.mats[i], hidden_acts_trans.mats[i]);

        if (i < rl->steps - 1) {
            mat_dot(delta_hidden.mats[i], weights_output_trans, delta_output.mats[i]);
            mat_dot(temp, weights_hidden_trans, delta_hidden.mats[i + 1]);
            mat_add(delta_hidden.mats[i], delta_hidden.mats[i], temp);
            mat_had(delta_hidden.mats[i], delta_hidden.mats[i], lins_deriv_hidden.mats[i]);
        }
        else {
            mat_dot(delta_hidden.mats[i], weights_output_trans, delta_output.mats[i]);
            mat_had(delta_hidden.mats[i], delta_hidden.mats[i], lins_deriv_hidden.mats[i]);
        }

        mat_dot(tens3D_grad_out->mats[i], weights_input_trans, delta_hidden.mats[i]);

        mat_dot(dw_input_current, delta_hidden.mats[i], input_trans.mats[i]);

        if (i > 0) {
            mat_dot(dw_hidden_current, delta_hidden.mats[i], hidden_acts_trans.mats[i]);
        }
        else {
            mat broadcasted = mat_alloc(rl->batch_size, rl->hidden_size);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int j = 0; j < rl->batch_size; ++j) {
                for (int k = 0; k < rl->hidden_size; ++k) {
                    mat_at(broadcasted, j, k) = mat_at(initial_hidden_trans, 0, k);
                }
            }

            mat_dot(dw_hidden_current, delta_hidden.mats[i], broadcasted);

            free(broadcasted.vals);

            mat_dot(temp, weights_hidden_trans, delta_hidden.mats[0]);

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < rl->hidden_size; ++j) {
                float sum = 0.0f;

                for (int k = 0; k < rl->batch_size; ++k) {
                    sum += mat_at(temp, j, k);
                }

                mat_at(dinitial_hidden, j, 0) = sum;
            }
        }

        mat_add(dw_input, dw_input, dw_input_current);
        mat_add(dw_hidden, dw_hidden, dw_hidden_current);
        mat_add(dw_output, dw_output, dw_output_current);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < rl->hidden_size; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < rl->batch_size; ++k) {
                sum += mat_at(delta_hidden.mats[i], j, k);
            }

            mat_at(db_hidden, j, 0) = sum;
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < rl->output_size; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < rl->batch_size; ++k) {
                sum += mat_at(delta_output.mats[i], j, k);
            }

            mat_at(db_output, j, 0) = sum;
        }
    }

    *grad_out = tens3D_grad_out;

    mat_scale(dw_input, dw_input, 1.0f / rl->batch_size);
    mat_func(dw_input, dw_input, clip);
    mat_scale(dw_input, dw_input, rate);
    mat_sub(rl->weights_input, rl->weights_input, dw_input);

    mat_scale(dw_hidden, dw_hidden, 1.0f / rl->batch_size);
    mat_func(dw_hidden, dw_hidden, clip);
    mat_scale(dw_hidden, dw_hidden, rate);
    mat_sub(rl->weights_hidden, rl->weights_hidden, dw_hidden);

    mat_scale(dw_output, dw_output, 1.0f / rl->batch_size);
    mat_func(dw_output, dw_output, clip);
    mat_scale(dw_output, dw_output, rate);
    mat_sub(rl->weights_output, rl->weights_output, dw_output);

    mat_scale(db_hidden, db_hidden, 1.0f / rl->batch_size);
    mat_func(db_hidden, db_hidden, clip);
    mat_scale(db_hidden, db_hidden, rate);
    mat_sub(rl->biases_hidden, rl->biases_hidden, db_hidden);

    mat_scale(db_output, db_output, 1.0f / rl->batch_size);
    mat_func(db_output, db_output, clip);
    mat_scale(db_output, db_output, rate);
    mat_sub(rl->biases_output, rl->biases_output, db_output);

    mat_scale(dinitial_hidden, dinitial_hidden, 1.0f / rl->batch_size);
    mat_func(dinitial_hidden, dinitial_hidden, clip);
    mat_scale(dinitial_hidden, dinitial_hidden, rate);
    mat_sub(rl->initial_hidden, rl->initial_hidden, dinitial_hidden);

    tens3D_destroy(delta_hidden);
    tens3D_destroy(delta_output);
    tens3D_destroy(input_trans);
    tens3D_destroy(hidden_acts_trans);
    free(initial_hidden_trans.vals);
    tens3D_destroy(lins_deriv_hidden);
    tens3D_destroy(lins_deriv_output);
    free(weights_input_trans.vals);
    free(weights_hidden_trans.vals);
    free(weights_output_trans.vals);
    free(temp.vals);
    free(dw_input_current.vals);
    free(dw_hidden_current.vals);
    free(dw_output_current.vals);
    free(dw_input.vals);
    free(dw_hidden.vals);
    free(dw_output.vals);
    free(db_hidden.vals);
    free(db_output.vals);
    free(dinitial_hidden.vals);
}

void recurrent_destroy(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    free(rl->weights_input.vals);
    free(rl->weights_hidden.vals);
    free(rl->weights_output.vals);
    free(rl->biases_hidden.vals);
    free(rl->biases_output.vals);
    free(rl->initial_hidden.vals);

    tens3D_destroy(rl->input_cache);
    tens3D_destroy(rl->lins_cache_hidden);
    tens3D_destroy(rl->acts_cache_hidden);
    tens3D_destroy(rl->lins_cache_output);

    free(rl);
}

void recurrent_he(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_normal(rl->weights_input, 0, sqrt(2.0f / rl->input_size));
    mat_normal(rl->weights_hidden, 0, sqrt(2.0f / rl->hidden_size));
    mat_normal(rl->weights_output, 0, sqrt(2.0f / rl->hidden_size));
    mat_fill(rl->biases_hidden, 0);
    mat_fill(rl->biases_output, 0);
    mat_fill(rl->initial_hidden, 0);
}

void recurrent_glorot(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_normal(rl->weights_input, 0, sqrt(2.0f / (rl->input_size + rl->hidden_size)));
    mat_normal(rl->weights_hidden, 0, sqrt(2.0f / (rl->hidden_size + rl->hidden_size)));
    mat_normal(rl->weights_output, 0, sqrt(2.0f / (rl->hidden_size + rl->output_size)));
    mat_fill(rl->biases_hidden, 0);
    mat_fill(rl->biases_output, 0);
    mat_fill(rl->initial_hidden, 0);
}

void recurrent_print(layer l)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_print(rl->weights_input);
    mat_print(rl->weights_hidden);
    mat_print(rl->weights_output);
    mat_print(rl->biases_hidden);
    mat_print(rl->biases_output);
    mat_print(rl->initial_hidden);
}

void recurrent_save(layer l, FILE *f)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_save(rl->weights_input, f);
    mat_save(rl->weights_hidden, f);
    mat_save(rl->weights_output, f);
    mat_save(rl->biases_hidden, f);
    mat_save(rl->biases_output, f);
    mat_save(rl->initial_hidden, f);
}

void recurrent_load(layer l, FILE *f)
{
    recurrent_layer *rl = (recurrent_layer *)l.data;

    mat_load(rl->weights_input, f);
    mat_load(rl->weights_hidden, f);
    mat_load(rl->weights_output, f);
    mat_load(rl->biases_hidden, f);
    mat_load(rl->biases_output, f);
    mat_load(rl->initial_hidden, f);
}
