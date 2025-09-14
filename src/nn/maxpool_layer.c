#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer maxpool_layer_alloc(int x_r, int x_c, int x_d,
                          int x_b, int pooling_r, int pooling_c)
{
    assert(x_r % pooling_r == 0);
    assert(x_c % pooling_c == 0);

    maxpool_layer *ml = malloc(sizeof(maxpool_layer));

    ml->x_r = x_r;
    ml->x_c = x_c;
    ml->x_d = x_d;
    ml->x_b = x_b;

    ml->pooling_r = pooling_r;
    ml->pooling_c = pooling_c;

    ml->y_r = x_r / pooling_r;
    ml->y_c = x_c / pooling_c;

    ml->mask = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = ml;

    l.forward = maxpool_forward;
    l.backprop = maxpool_backprop;
    l.destroy = maxpool_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void maxpool_forward(layer l, tens x, tens *y)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;

    assert(x.dims[R] == ml->x_r);
    assert(x.dims[C] == ml->x_c);
    assert(x.dims[D] == ml->x_d);
    assert(x.dims[B] == ml->x_b);

    *y = tens_alloc(ml->y_r, ml->y_c, ml->x_d, ml->x_b);

    tens_fill(ml->mask, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->x_b; ++i) {
        for (int j = 0; j < ml->x_d; ++j) {
            for (int k = 0; k < ml->y_r; ++k) {
                for (int l = 0; l < ml->y_c; ++l) {
                    float max = -FLT_MAX;
                    int max_r = 0;
                    int max_c = 0;

                    for (int m = 0; m < ml->pooling_r; ++m) {
                        for (int n = 0; n < ml->pooling_c; ++n) {
                            int r = k * ml->pooling_r + m;
                            int c = l * ml->pooling_c + n;
                            float val = tens_at(x, r, c, j, i);

                            if (val > max) {
                                max = val;
                                max_r = m;
                                max_c = n;
                            }
                        }
                    }

                    tens_at(*y, k, l, j, i) = max;

                    int mask_r = k * ml->pooling_r + max_r;
                    int mask_c = l * ml->pooling_c + max_c;

                    tens_at(ml->mask, mask_r, mask_c, j, i) = 1.0f;
                }
            }
        }
    }
}

void maxpool_backprop(layer l, tens dy, tens *dx, float rate)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;

    assert(dy.dims[R] == ml->y_r);
    assert(dy.dims[C] == ml->y_c);
    assert(dy.dims[D] == ml->x_d);
    assert(dy.dims[B] == ml->x_b);

    *dx = tens_alloc(ml->x_r, ml->x_c, ml->x_d, ml->x_b);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ml->x_b; ++i) {
        for (int j = 0; j < ml->x_d; ++j) {
            for (int k = 0; k < ml->y_r; ++k) {
                for (int l = 0; l < ml->y_c; ++l) {
                    for (int m = 0; m < ml->pooling_r; ++m) {
                        for (int n = 0; n < ml->pooling_c; ++n) {
                            int r = k * ml->pooling_r + m;
                            int c = l * ml->pooling_c + n;

                            if (tens_at(ml->mask, r, c, j, i) == 1.0f) {
                                float val = tens_at(dy, k, l, j, i);
                                tens_at(*dx, r, c, j, i) = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

void maxpool_destroy(layer l)
{
    maxpool_layer *ml = (maxpool_layer *)l.data;

    tens_destroy(ml->mask);

    free(ml);
}
