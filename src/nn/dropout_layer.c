#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"
#include "utils.h"

layer dropout_layer_alloc(int x_r, int x_c,
                          int x_d, int x_b, float rate)
{
    dropout_layer *dl = malloc(sizeof(dropout_layer));

    dl->x_r = x_r;
    dl->x_c = x_c;
    dl->x_d = x_d;
    dl->x_b = x_b;

    dl->rate = rate;

    dl->mask = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = dl;

    l.forward = dropout_forward;
    l.backprop = dropout_backprop;
    l.destroy = dropout_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void dropout_forward(layer l, tens x, tens *y)
{
    dropout_layer *dl = (dropout_layer *)l.data;


    assert(x.dims[R] == dl->x_r);
    assert(x.dims[C] == dl->x_c);
    assert(x.dims[D] == dl->x_d);
    assert(x.dims[B] == dl->x_b);

    *y = tens_alloc(dl->x_r, dl->x_c, dl->x_d, dl->x_b);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dl->x_b; ++i) {
        for (int j = 0; j < dl->x_d; ++j) {
            for (int k = 0; k < dl->x_r; ++k) {
                for (int l = 0; l < dl->x_c; ++l) {
                    tens_at(dl->mask, k, l, j, i) =
                        rand_float(0.0f, 1.0f) > dl->rate ? 1.0f : 0.0f;
                }
            }
        }
    }

    tens_had(*y, x, dl->mask);
}

void dropout_backprop(layer l, tens dy, tens *dx, float rate)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    assert(dy.dims[R] == dl->x_r);
    assert(dy.dims[C] == dl->x_c);
    assert(dy.dims[D] == dl->x_d);
    assert(dy.dims[B] == dl->x_b);

    *dx = tens_alloc(dl->x_r, dl->x_c, dl->x_d, dl->x_b);

    tens_had(*dx, dy, dl->mask);
}

void dropout_destroy(layer l)
{
    dropout_layer *dl = (dropout_layer *)l.data;

    tens_destroy(dl->mask);

    free(dl);
}
