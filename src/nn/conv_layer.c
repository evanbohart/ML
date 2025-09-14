#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "nn.h"

layer conv_layer_alloc(int x_r, int x_c, int x_d,
                       int x_b, int w_r, int w_c,
                       int convolutions, int stride, int x_padding[4])
{
    conv_layer *cl = malloc(sizeof(conv_layer));

    int y_r = (x_r + x_padding[0] + x_padding[1] -
               w_r) / stride + 1;
    int y_c = (x_c + x_padding[2] + x_padding[3] -
               w_c) / stride + 1;

    cl->x_r = x_r;
    cl->x_c = x_c;
    cl->x_d = x_d;
    cl->x_b = x_b;

    cl->y_r = y_r;
    cl->y_c = y_c;
    cl->convolutions = convolutions;

    cl->w_r = w_r;
    cl->w_c = w_c;
    cl->stride = stride;

    memcpy(cl->x_padding, x_padding, sizeof(x_padding));

    int dy_padding[4] = { w_r - 1 - x_padding[TOP],
                          w_r - 1 - x_padding[BOTTOM],
                          w_c - 1 - x_padding[LEFT],
                          w_c - 1 - x_padding[RIGHT] };
    memcpy(cl->dy_padding, dy_padding, sizeof(dy_padding));

    cl->w = tens_alloc(w_r, w_c, x_d, convolutions);
    cl->b = tens_alloc(y_r, y_c, convolutions, 1);

    cl->x_padded = tens_alloc(x_r + x_padding[TOP] + x_padding[BOTTOM],
                              x_c + x_padding[LEFT] + x_padding[RIGHT],
                              x_d, x_b);
    cl->dy_padded = tens_alloc(y_r + dy_padding[TOP] + dy_padding[BOTTOM],
                               y_c + dy_padding[LEFT] + dy_padding[RIGHT],
                               convolutions, x_b);
    cl->w_180 = tens_alloc(w_r, w_r, x_d, convolutions);

    cl->dw = tens_alloc(w_r, w_r, x_d, convolutions);
    cl->db = tens_alloc(y_r, y_c, convolutions, 1);

    layer l;

    l.data = cl;

    l.forward = conv_forward;
    l.backprop = conv_backprop;
    l.destroy = conv_destroy;

    l.init = conv_init;
    l.print = conv_print;
    l.save = conv_save;
    l.load = conv_load;

    return l;
}

void conv_forward(layer l, tens x, tens *y)
{
    conv_layer *cl = (conv_layer *)l.data;

    assert(x.dims[R] == cl->x_r);
    assert(x.dims[C] == cl->x_c);
    assert(x.dims[D] == cl->x_d);
    assert(x.dims[B] == cl->x_b);

    *y = tens_alloc(cl->y_r, cl->y_c, cl->convolutions, cl->x_b);

    tens_pad(cl->x_padded, x, cl->x_padding);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->x_b; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            for (int k = 0; k < cl->y_r; ++k) {
                for (int l = 0; l < cl->y_c; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < cl->x_d; ++m) {
                        for (int n = 0; n < cl->w_r; ++n) {
                            for (int o = 0; o < cl->w_c; ++o) {
                                float x_val = tens_at(cl->x_padded, k + n, l + o, m, i);
                                float w_val = tens_at(cl->w, n, o, m, j);
                                sum += x_val * w_val;
                            }
                        }
                    }

                    tens_at(*y, k, l, j, i) = sum;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->x_b; ++i) {
        for (int j = 0; j < cl->convolutions; ++j) {
            for (int k = 0; k < cl->y_r; ++k) {
                for (int l = 0; l < cl->y_c; ++l) {
                    tens_at(*y, k, l, j, i) += tens_at(cl->b, k, l, j, 0);
                }
            }
        }
    }
}

void conv_backprop(layer l, tens dy, tens *dx, float rate)
{
    conv_layer *cl = (conv_layer *)l.data;

    assert(dy.dims[R] == cl->y_r);
    assert(dy.dims[C] == cl->y_c);
    assert(dy.dims[D] == cl->convolutions);
    assert(dy.dims[B] == cl->x_b);

    *dx = tens_alloc(cl->x_r, cl->x_c, cl->x_d, cl->x_b);

    tens_pad(cl->dy_padded, dy, cl->dy_padding);

    int flip[4] = { 1, 1, 0, 0 };
    tens_180(cl->w_180, cl->w, flip);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->x_b; ++i) {
        for (int j = 0; j < cl->x_d; ++j) {
            for (int k = 0; k < cl->x_r; ++k) {
                for (int l = 0; l < cl->x_c; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < cl->convolutions; ++m) {
                        for (int n = 0; n < cl->w_r; ++n) {
                            for (int o = 0; o < cl->w_c; ++o) {
                                float dy_val = tens_at(cl->dy_padded, k + n, l + o, m, i);
                                float w_180_val = tens_at(cl->w_180, n, o, j, i);

                                sum += dy_val * w_180_val;
                            }
                        }
                    }

                    tens_at(*dx, k, l, j, i) = sum;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->x_d; ++j) {
            for (int k = 0; k < cl->w_r; ++k) {
                for (int l = 0; l < cl->w_r; ++l) {
                    float sum = 0.0f;

                    for (int m = 0; m < cl->x_b; ++m) {
                        for (int n = 0; n < cl->y_r; ++n) {
                            for (int o = 0; o < cl->y_c  ; ++o) {
                                float x_val = tens_at(cl->x_padded, k + n, k + o, j, m);
                                float dy_val = tens_at(dy, n, o, i, m);

                                sum += x_val * dy_val;
                            }
                        }
                    }

                    tens_at(cl->dw, k, l, j, i) = sum;
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cl->convolutions; ++i) {
        for (int j = 0; j < cl->y_r; ++j) {
            for (int k = 0; k < cl->y_c; ++k) {
                float sum = 0;

                for (int l = 0; l < cl->x_b; ++l) {
                    sum += tens_at(dy, j, k, i, l);
                }

                tens_at(cl->db, j, k, i, 0) = sum;
            }
        }
    }

    tens_scale(cl->dw, cl->dw, rate / cl->x_b);
    tens_func(cl->dw, cl->dw, clip);
    tens_sub(cl->w, cl->w, cl->dw);

    tens_scale(cl->db, cl->db, rate / cl->x_b);
    tens_func(cl->db, cl->db, clip);
    tens_sub(cl->b, cl->b, cl->db);
}

void conv_destroy(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_destroy(cl->w);
    tens_destroy(cl->b);

    tens_destroy(cl->x_padded);
    tens_destroy(cl->dy_padded);
    tens_destroy(cl->w_180);

    tens_destroy(cl->dw);
    tens_destroy(cl->db);

    free(cl);
}

void conv_init(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_normal(cl->w, 0, sqrt(2.0 / (cl->x_d * cl->w_r * cl->w_r)));
    tens_fill(cl->b, 0);
}

void conv_print(layer l)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_print(cl->w);
    tens_print(cl->b);
}

void conv_save(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_save(cl->w, f);
    tens_save(cl->b, f);
}

void conv_load(layer l, FILE *f)
{
    conv_layer *cl = (conv_layer *)l.data;

    tens_load(cl->w, f);
    tens_load(cl->b, f);
}
