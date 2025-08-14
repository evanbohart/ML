#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

nn nn_alloc(int max_nodes)
{
    nn n;

    n.max_nodes = max_nodes;
    n.num_nodes = 0;
    n.nodes = malloc(max_nodes * sizeof(node));

    return n;
}

void nn_add_layer(nn *n, layer l)
{
    assert(n->num_nodes != n->max_nodes);

    node d;

    d.type = LAYER;
    d.l = l;

    n->nodes[n->num_nodes++] = d;
}

void nn_add_block(nn *n, block b)
{
    assert(n->num_nodes != n->max_nodes);

    node d;

    d.type = BLOCK;
    d.b = b;

    n->nodes[n->num_nodes++] = d;
}

void nn_forward(nn n, tens x, tens *y)
{
    tens x_current = x;
    tens y_current;

    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                n.nodes[i].l.forward(n.nodes[i].l, x_current, &y_current);
                break;
            case BLOCK:
                n.nodes[i].b.forward(n.nodes[i].b, x_current, &y_current);
                break;
        }

        if (i > 0) {
            switch (x_current.type) {
                case MAT:
                    free(x_current.m.vals);
                    break;
                case TENS3D:
                    tens3D_destroy(x_current.t3);
                    break;
                case TENS4D:
                    tens4D_destroy(x_current.t4);
                    break;
            }
        }

        x_current = y_current;
    }

    *y = y_current;
}

void nn_backprop(nn n, tens dy, tens *dx, float rate)
{
    tens dy_current = dy;
    tens dx_current;

    for (int i = n.num_nodes - 1; i >= 0; --i) {
        switch (n.nodes[i].type) {
            case LAYER:
                n.nodes[i].l.backprop(n.nodes[i].l, dy_current, &dx_current, rate);
                break;
            case BLOCK:
                n.nodes[i].b.backprop(n.nodes[i].b, dy_current, &dx_current, rate);
                break;
        }

        if (i < n.num_nodes - 1) {
            switch (dy_current.type) {
                case MAT:
                    free(dy_current.m.vals);
                    break;
                case TENS3D:
                    tens3D_destroy(dy_current.t3);
                    break;
                case TENS4D:
                    tens4D_destroy(dy_current.t4);
                    break;
            }
        }

        dy_current = dx_current;
    }

    *dx = dx_current;
}

void nn_destroy(nn n)
{
    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                n.nodes[i].l.destroy(n.nodes[i].l);
                break;
            case BLOCK:
                n.nodes[i].b.destroy(n.nodes[i].b);
                break;
        }
    }
}

void nn_init(nn n)
{
    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                if (n.nodes[i].l.init) {
                    n.nodes[i].l.init(n.nodes[i].l);
                }

                break;
            case BLOCK:
                if (n.nodes[i].b.init) {
                    n.nodes[i].b.init(n.nodes[i].b);
                }

                break;
        }
    }
}

void nn_print(nn n)
{
    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                if (n.nodes[i].l.print) {
                    n.nodes[i].l.print(n.nodes[i].l);
                }

                break;
            case BLOCK:
                if (n.nodes[i].b.print) {
                    n.nodes[i].b.print(n.nodes[i].b);
                }

                break;
        }
    }
}

void nn_save(nn n, FILE *f)
{
    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                if (n.nodes[i].l.save) {
                    n.nodes[i].l.save(n.nodes[i].l, f);
                }

                break;
            case BLOCK:
                if (n.nodes[i].b.save) {
                    n.nodes[i].b.save(n.nodes[i].b, f);
                }

                break;
        }
    }
}

void nn_load(nn n, FILE *f)
{
    for (int i = 0; i < n.num_nodes; ++i) {
        switch (n.nodes[i].type) {
            case LAYER:
                if (n.nodes[i].l.load) {
                    n.nodes[i].l.load(n.nodes[i].l, f);
                }

                break;
            case BLOCK:
                if (n.nodes[i].b.load) {
                    n.nodes[i].b.load(n.nodes[i].b, f);
                }

                break;
        }
    }
}
