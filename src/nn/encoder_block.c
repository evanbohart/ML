#include <stdlib.h>
#include <assert.h>
#include "nn.h"

block encoder_block_alloc(int sub_layers, int seq_len, int d_model,
                          int d_k, int d_ff, int batch_size)
{
    assert(d_model % d_k == 0);

    encoder_block *eb = malloc(sizeof(encoder_block));

    eb->sub_layers = sub_layers;
    eb->seq_len = seq_len;
    eb->d_model = d_model;
    eb->d_k = d_k;
    eb->h_size = d_model / d_k;
    eb->d_ff = d_ff;
    eb->batch_size = batch_size;

    eb->a_skip = tens4D_alloc(seq_len, d_model, batch_size, sub_layers);
    eb->ff_skip = tens4D_alloc(seq_len, d_model, batch_size, sub_layers);

    eb->attention_layers = malloc(sub_layers * sizeof(layer));
    eb->attention_layernorm_layers = malloc(sub_layers * sizeof(layer));
    eb->mlp_hidden_layers = malloc(sub_layers * sizeof(layer));
    eb->relu_layers = malloc(sub_layer * sizeof(layer));
    eb->mlp_output_layers = malloc(sub_layers * sizeof(layer));
    eb->mlp_layernorm_layers = malloc(sub_layers * sizeof(layer));

    for (int i = 0; i < sub_layers; ++i) {
        eb->attention_layers[i] = attention_layer_alloc(d_model, seq_len,
                                                        d_k, batch_size);
        eb->attention_layernorm_layers[i] = layernorm_layer_3D_alloc(d_model, seq_len, batch_size);
        eb->mlp_hidden_layers[i] = dense_layer_alloc(d_model, d_ff, seq_len);
        eb->relu_layers[i] = relu_layer_alloc_3D(d_ff, seq_len, batch_size);
        eb->mlp_output_layers[i] = dense_layer_alloc(d_ff, d_model, seq_len);
        eb->mlp_layernorm_layers[i] = layernorm_layer_2D_alloc(d_model, seq_len)
    }

    block b;

    b.data = eb;

    b.forward = encoder_forward;
    b.backprop = encoder_backprop;
    b.destroy = encoder_destroy;

    b.init = encoder_init;
    b.print = encoder_print;
    b.save = encoder_save;
    b.load = encoder_load;

    return b;
}

void encoder_forward(block b, tens x, tens *y)
{
    encoder_block *eb = (encoder_block *)b.data;

    assert(x.type == TENS3D);
    assert(x.t3.rows == eb->seq_len);
    assert(x.t3.cols == eb->d_model);
    assert(x.t3.depth == eb->batch_size);

    y->type = TENS3D;
    y->t3.rows = eb->seq_len;
    y->t3.cols = eb->d_model;
    y->t3.depth == eb->batch_size;

    tens x_current = x;
    tens y_current;

    for (int i = 0; i < eb->sub_layers; ++i) {
        tens3D_copy(eb->a_skip.tens3Ds[i], x_current.t3);

        eb->attention_layers[i].forward(eb->attention_layers[i], x_current, &y_current);

        tens3D_add(y_current.t3, y_current.t3, eb->a_skip.tens3Ds[i]);

        x_current = y_current;

        eb->attention_layernorm_layers[i].forward(eb->attention_layernorm_layers[i], x_current, &y_current);

        x_current = y_current;
    }
}
