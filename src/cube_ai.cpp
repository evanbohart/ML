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
        root->value = 0;
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
        feed_forward(policy, inputs);

        root->children = new state*[12];
        for (int i = 0; i < 12; ++i) {
            root->children[i] = new state;
            root->children[i]->c.copy(root->c);
            root->children[i]->c.turn((model::move)i);
            root->children[i]->prior = mat_at(policy.acts[policy.layers - 2], i, 0);
            root->children[i]->visits = 0;
            root->children[i]->value = 0;
            root->children[i]->parent = root;
            root->children[i]->children = nullptr;
        }

        free(inputs.vals);
    }

    void tree::select_child(state *&root)
    {
        assert(root->children);

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

    double tree::uct(const state *root) const
    {
        assert(root->parent);
        return root->value + sqrt(log(root->parent->visits + 1) / (root->visits + 1));// + root->prior;
    }

    double tree::eval(net value, const state *root)
    {
        mat inputs = mat_alloc(mat_at(value.topology, 0, 0), 1);
        root->c.get_inputs(inputs);
        feed_forward(value, inputs);

        return mat_at(value.acts[value.layers - 2], 0, 0);
    }

    double tree::rollout(net policy, state *leaf)
    {
        model::cube c;
        c.copy(leaf->c);

        mat inputs = mat_alloc(48, 1);
        int n = 0;
        for (int i = 0; i < 100 && !c.is_solved(); ++i) {
            c.get_inputs(inputs);
            feed_forward(policy, inputs);

            double max = 0;
            int index = 0;
            for (int j = 0; j < 12; ++j) {
                if (mat_at(policy.acts[policy.layers - 2], j, 0) > max) {
                    max = mat_at(policy.acts[policy.layers - 2], j, 0);
                    index = j;
                }
            }

            c.turn((model::move)index);
            ++n;
        }

        return c.is_solved() * pow(.95, n);
    }

    void tree::backup(state *leaf, double value)
    {
        int n = 0;
        while (leaf) {
            ++leaf->visits;
            leaf->value = std::max(leaf->value, value * pow(.95, n));
            leaf = leaf->parent;
            ++n;
        }
    }

    void tree::mcts(net policy, int n)
    {
        expand_state(policy, root);

        for (int i = 0; i < n; ++i) {
            state *leaf = root;
            traverse(leaf);

            double val;
            if (leaf->c.is_solved()) {
                val = 1;
            }
            else {
                expand_state(policy, leaf);
                select_child(leaf);
                val = rollout(policy, leaf);
            }

            backup(leaf, val);
        }
    }

    void tree::generate_solution(stack<model::move> &moves, state *leaf)
    {
        assert(leaf->c.is_solved());

        while (leaf != root) {
            state *temp = leaf;
            leaf = leaf->parent;
            for (int i = 0; i < 12; ++i) {
                if (leaf->children[i] == temp) {
                    moves.push((model::move)i);
                    break;
                }
            }
        }
    }

    bool tree::solve(net value, net policy, stack<model::move> &moves, int n)
    {
        expand_state(policy, root);

        for (int i = 0; i < n; ++i) {
            state *leaf = root;
            traverse(leaf);

            if (leaf->c.is_solved()) {
                generate_solution(moves, leaf);
                return true;
            }
            else {
                expand_state(policy, leaf);
                select_child(leaf);
                backup(leaf, eval(value, leaf));
            }
        }

        return false;
    }

    void tree::train_value(net value, double rate)
    {
        assert(root->children);

        mat inputs = mat_alloc(48, 1);
        root->c.get_inputs(inputs);

        mat targets = mat_alloc(1, 1);
        mat_at(targets, 0, 0) = root->value;

        backprop(value, inputs, targets, rate);

        free(inputs.vals);
        free(targets.vals);
    }

    void tree::train_policy(net policy, double rate)
    {
        assert(root->children);

        mat inputs = mat_alloc(48, 1);
        root->c.get_inputs(inputs);

        mat targets = mat_alloc(12, 1);
        int max = 0;
        int index = 0;
        for (int i = 0; i < 12; ++i) {
            if (root->children[i]->visits > max) {
                max = root->children[i]->visits;
                index = i;
            }
        }

        mat_zero(targets);
        mat_at(targets, index, 0) = 1;

        backprop(policy, inputs, targets, rate);

        free(inputs.vals);
        free(targets.vals);
    }
}
