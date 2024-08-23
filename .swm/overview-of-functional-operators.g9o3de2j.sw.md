---
title: Overview of Functional Operators
---
# Overview of Functional Operators

Functional operators in <SwmToken path="tools/nightly.py" pos="306:1:1" line-data="    pytorch, platform = &quot;&quot;, &quot;&quot;">`pytorch`</SwmToken> are callables that do not alter the state of their inputs or any external state. They are designed to be non-mutating, meaning they do not change the value of their input, including both metadata and data for tensors. Additionally, functional operators have no side effects, ensuring that they do not change states visible from outside, such as module parameters.

# Example of Using Functional Operators

Here are some examples of how to use various functional operators in <SwmToken path="tools/nightly.py" pos="306:1:1" line-data="    pytorch, platform = &quot;&quot;, &quot;&quot;">`pytorch`</SwmToken>.

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/fold.h" line="87">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/fold.h" pos="89:4:4" line-data="/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));">`unfold`</SwmToken> function is used to extract sliding local blocks from a batched input tensor. It is called with the input tensor and options specifying the kernel size, padding, and stride.

````c
/// ```
/// namespace F = torch::nn::functional;
/// F::unfold(input, F::UnfoldFuncOptions({2, 2}).padding(1).stride(2));
/// ```
````

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/activation.h" line="455">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="457:4:4" line-data="/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));">`relu6`</SwmToken> function applies the <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="457:4:4" line-data="/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));">`relu6`</SwmToken> activation function to the input tensor. It is called with the input tensor and options specifying whether the operation should be performed in-place.

````c
/// ```
/// namespace F = torch::nn::functional;
/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));
````

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/activation.h" line="67">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="69:4:4" line-data="/// F::selu(input, F::SELUFuncOptions(false));">`selu`</SwmToken> function applies the SELU activation function to the input tensor. It is called with the input tensor and options specifying whether the operation should be performed in-place.

````c
/// ```
/// namespace F = torch::nn::functional;
/// F::selu(input, F::SELUFuncOptions(false));
````

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/activation.h" line="425">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="427:4:4" line-data="/// F::relu(x, F::ReLUFuncOptions().inplace(true));">`relu`</SwmToken> function applies the <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="427:4:4" line-data="/// F::relu(x, F::ReLUFuncOptions().inplace(true));">`relu`</SwmToken> activation function to the input tensor. It is called with the input tensor and options specifying whether the operation should be performed in-place.

````c
/// ```
/// namespace F = torch::nn::functional;
/// F::relu(x, F::ReLUFuncOptions().inplace(true));
````

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/fold.h" line="41">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/fold.h" pos="43:4:4" line-data="/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));">`fold`</SwmToken> function is used to combine an array of sliding local blocks into a large containing tensor. It is called with the input tensor and options specifying the output size, kernel size, padding, and stride.

````c
/// ```
/// namespace F = torch::nn::functional;
/// F::fold(input, F::FoldFuncOptions({3, 2}, {2, 2}));
````

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/conv.h" line="53">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/conv.h" pos="54:4:4" line-data="/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));">`conv1d`</SwmToken> function performs a 1D convolution operation on the input tensor. It is called with the input tensor, weight tensor, and options specifying the stride.

```c
/// namespace F = torch::nn::functional;
/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/activation.h" line="526">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="527:4:4" line-data="/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));">`celu`</SwmToken> function applies the CELU activation function to the input tensor. It is called with the input tensor and options specifying the alpha parameter and whether the operation should be performed in-place.

```c
/// namespace F = torch::nn::functional;
/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/activation.h" line="364">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/activation.h" pos="365:4:4" line-data="/// F::glu(input, GLUFuncOptions(1));">`glu`</SwmToken> function applies the GLU activation function to the input tensor. It is called with the input tensor and options specifying the dimension along which to split the input tensor.

```c
/// namespace F = torch::nn::functional;
/// F::glu(input, GLUFuncOptions(1));
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/loss.h" line="89">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/loss.h" pos="89:4:4" line-data="/// F::kl_div(input, target,">`kl_div`</SwmToken> function computes the Kullback-Leibler divergence loss between the input tensor and the target tensor. It is called with the input tensor, target tensor, and options specifying the reduction method and whether the target is in log space.

```c
/// F::kl_div(input, target,
/// F::KLDivFuncOptions.reduction(torch::kNone).log_target(false));
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/api/include/torch/nn/functional/conv.h" line="191">

---

The <SwmToken path="torch/csrc/api/include/torch/nn/functional/conv.h" pos="192:4:4" line-data="/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));">`conv_transpose1d`</SwmToken> function performs a 1D transposed convolution operation on the input tensor. It is called with the input tensor, weight tensor, and options specifying the stride.

```c
/// namespace F = torch::nn::functional;
/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
```

---

</SwmSnippet>

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
