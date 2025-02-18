#include "cube.h"
#include <cassert>
#include <cmath>

#include <stdio.h>

tree::tree(const cube &c)
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
    if (root->children) {
        for (int i = 0; i < 12; ++i) {
            destroy(root->children[i]);
        }

        delete [] root->children;
    }

    delete root;
}

void tree::expand_state(cnet cn, net policy, state *root)
{
    assert(!root->children);

    tens conv_inputs = tens_alloc(8, 6, 1);
    mat dense_inputs = mat_alloc(32, 1);

    root->c.get_inputs(conv_inputs);
    cnet_forward(cn, conv_inputs);

    tens_flatten(dense_inputs, cn.acts[cn.layers - 2]);
    net_forward(policy, dense_inputs);

    tens_destroy(&conv_inputs);
    free(dense_inputs.vals);

    root->children = new state*[12];
    for (int i = 0; i < 12; ++i) {
        root->children[i] = new state;
        root->children[i]->c.copy(root->c);
        root->children[i]->c.turn((move)i);
        root->children[i]->prior = mat_at(policy.acts[policy.layers - 2], i, 0);
        root->children[i]->visits = 0;
        root->children[i]->value = 0;
        root->children[i]->parent = root;
        root->children[i]->children = nullptr;
    }
}

void tree::select_child(state *&root)
{
    assert(root->children);

    double max = 0;
    int index = 0;
    for (int i = 0; i < 12; ++i) {
        if (uct(root->children[i]) > max) {
            max = uct(root->children[i]);
            index = i;
        }
    }

    root = root->children[index];
}

void tree::traverse(state *&root)
{
    assert(root->children);

    while (root->children) {
        select_child(root);
    }
}

double tree::uct(const state *root) const
{
    assert(root->parent);
    return root->value + sqrt(2) * sqrt(log(root->parent->visits + 1) / (root->visits + 1)) + root->prior;
}

double tree::eval(cnet cn, net value, state *root)
{
    tens conv_inputs = tens_alloc(8, 6, 1);
    root->c.get_inputs(conv_inputs);
    cnet_forward(cn, conv_inputs);

    mat dense_inputs = mat_alloc(32, 1);
    tens_flatten(dense_inputs, cn.acts[cn.layers - 2]);
    net_forward(value, dense_inputs);

    tens_destroy(&conv_inputs);
    free(dense_inputs.vals);

    return mat_at(value.acts[value.layers - 2], 0, 0);
}

void tree::backup(state *leaf, double val)
{
    int n = 0;
    while (leaf) {
        ++leaf->visits;
        leaf->value = std::max(leaf->value, val * pow(.95, n));
        leaf = leaf->parent;
        ++n;
    }
}

void tree::generate_solution(stack<move> &solution, state *leaf)
{
    assert(leaf->c.is_solved());

    while (leaf != root) {
        state *temp = leaf;
        leaf = leaf->parent;
        for (int i = 0; i < 12; ++i) {
            if (leaf->children[i] == temp) {
                solution.push((move)i);
                break;
            }
        }
    }
}

int tree::mcts(cnet cn, net policy, stack<move> &solution, int n)
{
    assert(!root->children);

    expand_state(cn, policy, root);

    for (int i = 0; i < n; ++i) {
        state *leaf = root;
        traverse(leaf);

        if (leaf->c.is_solved()) {
            generate_solution(solution, leaf);
            return i + 1;
        }
        else {
            expand_state(cn, policy, leaf);
            select_child(leaf);

            if (leaf->c.is_solved()) {
                generate_solution(solution, leaf);
                return i + 1;
            }

            backup(leaf, 0);
            }
    }

    return 0;
}

int tree::mcts(cnet cn, net value, net policy, stack<move> &solution, int n)
{
    assert(!root->children);

    expand_state(cn, policy, root);

    for (int i = 0; i < n; ++i) {
        state *leaf = root;
        traverse(leaf);

        if (leaf->c.is_solved()) {
            generate_solution(solution, leaf);
            return i + 1;
        }
        else {
            expand_state(cn, policy, leaf);
            select_child(leaf);

            if (leaf->c.is_solved()) {
                    generate_solution(solution, leaf);
                    return i + 1;
            }

            backup(leaf, eval(cn, value, leaf));
        }
    }

    return 0;
}

void tree::train_value(cnet cn, net value, stack<move> &solution, double rate)
{
    train_value(cn, value, solution, rate, root);
}

void tree::train_value(cnet cn, net value, stack<move> &solution, double rate, state *root)
{
    tens conv_inputs = tens_alloc(8, 6, 1);
    mat dense_inputs = mat_alloc(32, 1);
    mat targets = mat_alloc(1, 1);
    mat delta = mat_alloc(32, 1);

    root->c.get_inputs(conv_inputs);

    cnet_forward(cn, conv_inputs);
    tens_flatten(dense_inputs, cn.acts[cn.layers - 2]);

    mat_at(targets, 0, 0) = pow(.95, solution.size() - 1);

    net_backprop(value, dense_inputs, targets, rate, delta);
    cnet_backprop(cn, conv_inputs, delta, rate);

    tens_destroy(&conv_inputs);
    free(dense_inputs.vals);
    free(targets.vals);
    free(delta.vals);

    if (solution.size() == 0) return;

    state *next = root->children[solution.top()];
    solution.pop();

    train_value(cn, value, solution, rate *.95, next);
}

void tree::train_policy(cnet cn, net policy, stack<move> &solution, double rate)
{
    train_policy(cn, policy, solution, rate, root);
}

void tree::train_policy(cnet cn, net policy, stack<move> &solution, double rate, state *root)
{
    if (solution.size() == 0) return;

    tens conv_inputs = tens_alloc(8, 6, 1);
    mat dense_inputs = mat_alloc(32, 1);
    mat targets = mat_alloc(12, 1);
    mat delta = mat_alloc(32, 1);

    root->c.get_inputs(conv_inputs);

    cnet_forward(cn, conv_inputs);
    tens_flatten(dense_inputs, cn.acts[cn.layers - 2]);

    mat_fill(targets, 0);
    mat_at(targets, solution.top(), 0) = 1;

    net_backprop(policy, dense_inputs, targets, rate, delta);
    cnet_backprop(cn, conv_inputs, delta, rate);

    tens_destroy(&conv_inputs);
    free(dense_inputs.vals);
    free(targets.vals);
    free(delta.vals);

    state *next = root->children[solution.top()];
    solution.pop();

    train_policy(cn, policy, solution, rate * .95, next);
}
