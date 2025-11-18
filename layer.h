#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <memory>
#include <vector>
#include "value.h"
#include "neuron.h"

using std::shared_ptr;
using std::string;
using std::vector;

class layer {
    private:
      int nin;
      int nout;
      string activation;
      vector<shared_ptr<neuron>> neurons;
    public:
      layer(int nin, int nout, string activation);
      vector<shared_ptr<value>> forward(const vector<shared_ptr<value>>& x);
      vector<vector<shared_ptr<value>>> getParameters();
};
#endif