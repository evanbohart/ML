#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer softmax_layer_alloc(int x_r, int x_c,
                          int x_d, int x_b)
{
    softmax_layer *sl = malloc(sizeof(softmax_layer));

    sl->x_r = x_r;
    sl->x_c = x_c;
    sl->x_d = x_d;
    sl->x_b = x_b;

    sl->y_cache = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = sl;

    l.forward = softmax_forward;
    l.backprop = softmax_backprop;
    l.destroy = softmax_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void softmax_forward(layer l, tens x, tens *y)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    assert(x.dims[R] == sl->x_r);
    assert(x.dims[C] == sl->x_c);
    assert(x.dims[D] == sl->x_d);
    assert(x.dims[B] == sl->x_b);

    *y = tens_alloc(sl->x_r, sl->x_c, sl->x_d, sl->x_b);

    tens_softmax(*y, x);

    tens_copy(sl->y_cache, *y);
}

void softmax_backprop(layer l, tens dy, tens *dx, float rate)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    assert(dy.dims[R] == sl->x_r);
    assert(dy.dims[C] == sl->x_c);
    assert(dy.dims[D] == sl->x_d);
    assert(dy.dims[B] == sl->x_b);

    *dx = tens_alloc(sl->x_r, sl->x_c, sl->x_d, sl->x_b);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < sl->x_b; ++i) {
        for (int j = 0; j < sl->x_d; ++j) {
            for (int k = 0; k < sl->x_c; ++k) {
                for (int l = 0; l < sl->x_r; ++l) {
                    float sum = 0.0f;

                    float i_val = tens_at(sl->y_cache, l, k, j, i);

                    for (int m = 0; m < sl->x_r; ++m) {
                        float j_val = tens_at(sl->y_cache, m, k, j, i);

                        if (l == m) {
                            sum += i_val * (1.0f - j_val);
                        }
                        else {
                            sum += -i_val * j_val;
                        }
                    }

                    tens_at(*dx, k, l, j, i) = sum;
                }
            }
        }
    }

    tens_had(*dx, *dx, dy);
}

void softmax_destroy(layer l)
{
    softmax_layer *sl = (softmax_layer *)l.data;

    tens_destroy(sl->y_cache);

    free(sl);
}
