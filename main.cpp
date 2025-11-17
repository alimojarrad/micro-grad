#include <iostream>
#include "value.h"
using namespace std;
int main() {
    auto x = std::make_shared<value>(2.0);
    auto y = x->tanh();  
    y->backward();
    std::cout << x << std::endl;
    std::cout << y << std::endl;

    return 0;
}