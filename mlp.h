#ifndef MLP_H
#define MLP_H

#include <vector>
#include <memory>
#include <string>
#include "layer.h"

using std::shared_ptr;
using std::string;
using std::vector;

class mlp {
private:
    vector<shared_ptr<layer>> layers;
public:
    mlp(int nin, const vector<int>& sizes, string activation);
    vector<shared_ptr<value>> forward(const vector<shared_ptr<value>>& x);
    vector<vector<vector<shared_ptr<value>>>> getParameters();
    vector<vector<shared_ptr<value>>> forward(const vector<vector<shared_ptr<value>>>& batch);
};

#endif
