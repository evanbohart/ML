#include "cube.h"
#include <cassert>
#include <cmath>

namespace ai
{
    tree::tree(const model::cube &c)
    {
        root = new state;
        root->c.copy(c);
        root->prior = 0;
        root->visits = 0;
        root->value_sum = 0;
        root->parent = nullptr;
        root->children = nullptr;
    }

    tree::~tree()
    {
        destroy(root);
    }

    void tree::destroy(state *root)
    {
        if (!root->children) {
            delete root;
            return;
        }

        for (int i = 0; i < 12; ++i) {
            destroy(root->children[i]);
        }
    }

    void tree::expand_state(net policy, state *root)
    {
        mat inputs = mat_alloc(48, 1);
        root->c.get_inputs(inputs);
        feed_forward(policy, inputs, relu);
        mat_softmax(policy.acts[policy.layers - 2], policy.acts[policy.layers - 2]);

        root->children = new state*[12];
        for (int i = 0; i < 12; ++i) {
            root->children[i] = new state;
            root->children[i]->c.copy(root->c);
            root->children[i]->c.turn((model::move)i);
            root->children[i]->prior = mat_at(policy.acts[policy.layers - 2], i, 0);
            root->children[i]->visits = 0;
            root->children[i]->value_sum = 0;
            root->children[i]->parent = root;
            root->children[i]->children = nullptr;
        }

        free(inputs.vals);
    }

    void tree::select_child(state *&root)
    {
         double max_uct = 0;
         int index = 0;
         for (int i = 0; i < 12; ++i) {
             if (uct(root->children[i]) > max_uct) {
                 max_uct = uct(root->children[i]);
                 index = i;
             }
         }

         root = root->children[index];
    }

    void tree::traverse(state *&root)
    {
        while (root->children) {
            select_child(root);
        }
    }

    double tree::get_value(const state *root) const
    {
        if (root->visits > 0) return root->value_sum / root->visits;

        return 0;
    }

    double tree::uct(const state *root) const
    {
        return get_value(root) + root->prior * sqrt(root->parent->visits) / (root->visits + 1);
    }

    double tree::eval(net value, const state *root)
    {
        mat inputs = mat_alloc(mat_at(value.topology, 0, 0), 1);
        root->c.get_inputs(inputs);
        feed_forward(value, inputs, relu);

        return mat_at(value.acts[value.layers - 2], 0, 0);
    }

    double tree::rollout(net policy, state *leaf)
    {
        model::cube c;
        c.copy(leaf->c);

        mat inputs = mat_alloc(48, 1);
        for (int i = 0; i < 100 && !c.is_solved(); ++i) {
            c.get_inputs(inputs);
            feed_forward(policy, inputs, relu);
            mat_softmax(policy.acts[policy.layers - 2], policy.acts[policy.layers - 2]);

            double max = 0;
            int index = 0;
            for (int j = 0; j < 12; ++j) {
                if (mat_at(policy.acts[policy.layers - 2], j, 0) > max) {
                    max = mat_at(policy.acts[policy.layers - 2], j, 0);
                    index = j;
                }
            }

            c.turn((model::move)index);
        }

        return c.is_solved();
    }

    void tree::backup(state *leaf, double value)
    {
        while (leaf) {
            ++leaf->visits;
            leaf->value_sum += value;
            leaf = leaf->parent;
        }
    }

    void tree::mcts(net policy, int n)
    {
        expand_state(policy, root);

        for (int i = 0; i < n; ++i) {
            state *leaf = root;
            traverse(leaf);

            double value;
            if (!leaf->c.is_solved()) {
                expand_state(policy, leaf);
                select_child(leaf);
                value = rollout(policy, leaf);
            }
            else {
                value = 1;
            }

            backup(leaf, value);
        }
    }

    void tree::train_value(net value, double rate)
    {
        assert(root->children);

        mat inputs = mat_alloc(mat_at(value.topology, 0, 0), 1);
        root->c.get_inputs(inputs);

        mat targets = mat_alloc(1, 1);
        mat_at(targets, 0, 0) = get_value(root);

        feed_forward(value, inputs, sig);
        backprop(value, inputs, targets, MSE, dsig, rate);

        free(inputs.vals);
        free(targets.vals);
    }

    void tree::train_policy(net policy, double rate)
    {
        assert(root->children);

        mat inputs = mat_alloc(48, 1);
        root->c.get_inputs(inputs);

        mat targets = mat_alloc(12, 1);
        for (int i = 0; i < 12; ++i) {
            mat_at(targets, i, 0) = root->children[i]->visits;
        }
        mat_softmax(targets, targets);

        feed_forward(policy, inputs, relu);
        mat_softmax(policy.acts[policy.layers - 2], policy.acts[policy.layers - 2]);
        backprop(policy, inputs, targets, XE, drelu, rate);

        free(inputs.vals);
        free(targets.vals);
    }
}
