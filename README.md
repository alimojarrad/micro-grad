# Tiny Autograd Engine in C++

A lightweight **automatic differentiation engine** implemented from scratch in modern C++. Inspired by [micrograd](https://github.com/karpathy/micrograd), this engine supports building computation graphs, performing backpropagation, and calculating gradients — all with intuitive operator overloading.

🚀 **Great for learning deep-learning internals and autodiff!**

---

## ✨ Features

- Reverse-mode automatic differentiation (backpropagation)
- Modern C++ design using `std::shared_ptr` for safe dynamic computation graph handling
- Operator overloading for arithmetic operations (`+`, `-`, `*`, `/`, `pow`)
- Support for basic activation functions (e.g. `tanh`)
- Differentiable graph with lazy backward evaluation
- Gradient propagation through flexible computation structures

---

## 📦 Example

Here's a quick example showing how to build and evaluate a graph:

```cpp
#include "value.h"

int main() {
    auto a = std::make_shared<value>(2.0);
    auto b = std::make_shared<value>(3.0);

    // Expression: c = (a * b) + a.tanh()
    auto c = a * b + a->tanh();

    c->backward();

    std::cout << c << std::endl;  // data and grad
    std::cout << a << std::endl;  // grad = ∂c/∂a
    std::cout << b << std::endl;  // grad = ∂c/∂b

    return 0;
}
