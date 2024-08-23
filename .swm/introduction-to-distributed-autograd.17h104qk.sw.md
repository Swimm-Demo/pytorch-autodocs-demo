---
title: Introduction to Distributed Autograd
---
# Introduction to Distributed Autograd

Distributed Autograd extends the autograd engine to work across multiple machines, enabling the computation of gradients in a distributed setting. This is essential for training large models that cannot fit on a single machine.

# Core Concepts

Autograd is a key component for <SwmToken path="tools/nightly.py" pos="306:1:1" line-data="    pytorch, platform = &quot;&quot;, &quot;&quot;">`pytorch`</SwmToken> performance, with most of its heavy lifting implemented in C++. It involves shuffling data between Python and C++ to ensure data is in a form convenient for manipulation from C++. For any key data type that autograd manipulates, there are two implementations: a C++ type and a Python object type.

<SwmSnippet path="/torch/csrc/distributed/autograd/autograd.cpp" line="11">

---

The <SwmToken path="torch/csrc/distributed/autograd/autograd.cpp" pos="11:2:2" line-data="void backward(">`backward`</SwmToken> function in distributed autograd retrieves the context for a given context ID, performs initial preprocessing, and computes dependencies locally. This function is crucial for initiating the distributed backward pass.

```c++
void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.autograd.backward");
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  try {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // FIXME: crashes if exception type is not RuntimeError
    TORCH_CHECK(false, e.what());
  }
}
```

---

</SwmSnippet>

# Forward-mode Automatic Differentiation

Forward-mode automatic differentiation is one of the methods used by Autograd to compute derivatives efficiently.

# Locally Disabling Gradient Computation

You can locally disable gradient computation using context managers like `torch.no_grad()` to improve performance when gradients are not needed.

# Custom Function Utilities

Autograd allows you to define custom autograd functions by subclassing `torch.autograd.Function` and implementing the <SwmToken path="benchmarks/instruction_counts/core/api.py" pos="36:1:1" line-data="    FORWARD = &quot;Forward&quot;">`FORWARD`</SwmToken> and <SwmToken path="torch/csrc/distributed/autograd/autograd.cpp" pos="11:2:2" line-data="void backward(">`backward`</SwmToken> methods.

# Main Functions of Autograd

The main functions of Autograd include `Function.forward` and `Function.backward`.

## Function.forward

The `Function.forward` method is responsible for defining the computation performed at every call. It takes as input a context object and a variable number of arguments, and it returns the output of the computation. This method is crucial for specifying the forward pass of a neural network layer or any other differentiable operation.

## Function.backward

The `Function.backward` method defines the gradient computation for the operation performed in the <SwmToken path="benchmarks/instruction_counts/core/api.py" pos="36:1:1" line-data="    FORWARD = &quot;Forward&quot;">`FORWARD`</SwmToken> method. It takes as input a context object and the gradient of the output with respect to some loss, and it returns the gradient of the input with respect to the same loss. This method is essential for backpropagation, allowing the network to learn by updating weights based on the computed gradients.

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
