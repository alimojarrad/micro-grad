#include <iostream>
#include <vector>
#include <memory>
#include "layer.h"
#include "mlp.h"   
#include "neuron.h" 
 
using namespace std;

int main() {
    mlp model(3, {4, 4, 1}, "relu");

    vector<shared_ptr<value>> sample = {
        make_shared<value>(1.0),
        make_shared<value>(-2.0),
        make_shared<value>(0.5)
    };

    vector<shared_ptr<value>> target = {
        make_shared<value>(1.0)
    };

    auto out = model.forward(sample);

    shared_ptr<value> loss = make_shared<value>(0.0);
    for (int i = 0; i < out.size(); i++) {
        auto diff = out[i] - target[i];
        loss = *loss + (diff * diff);
    }

    loss->backward();

    auto params = model.getParameters();

    double lr = 0.01;
    for (auto& layer_params : params) {
        for (auto& p : layer_params) {
            for (auto& w : p) {
                w->data -= lr * w->grad;
            }
        }
    }

    cout << "Loss: " << loss->item() << endl;

    return 0;
}