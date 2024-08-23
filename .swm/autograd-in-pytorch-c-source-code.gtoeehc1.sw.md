---
title: Autograd in PyTorch C++ Source Code
---
# Overview

Autograd is a key component of <SwmToken path="tools/nightly.py" pos="306:1:1" line-data="    pytorch, platform = &quot;&quot;, &quot;&quot;">`pytorch`</SwmToken> that handles automatic differentiation. It is implemented in C++ to optimize performance, with data structures and operations designed to be efficient in this language. This document explains the core concepts and functionalities of Autograd in the C++ source code (Csrc).

# Data Structures

For any key data type that Autograd manipulates, there are two implementations: a C++ type and a Python object type. For example, variables in Autograd have both <SwmToken path="test/dynamo/test_global.py" pos="185:3:3" line-data="        class Variable:">`Variable`</SwmToken> in <SwmPath>[torch/csrc/autograd/variable.h](torch/csrc/autograd/variable.h)</SwmPath> (the C++ type) and <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1401:1:1" line-data="  THPVariable* var = reinterpret_cast&lt;THPVariable*&gt;(_var);">`THPVariable`</SwmToken> in <SwmPath>[torch/csrc/autograd/python_variable.h](torch/csrc/autograd/python_variable.h)</SwmPath> (the Python type). <SwmToken path="test/dynamo/test_global.py" pos="185:3:3" line-data="        class Variable:">`Variable`</SwmToken> contains the payload of a variable, while <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1401:1:1" line-data="  THPVariable* var = reinterpret_cast&lt;THPVariable*&gt;(_var);">`THPVariable`</SwmToken> contains a <SwmToken path="torch/csrc/autograd/engine.cpp" pos="130:4:4" line-data="C10_DEFINE_TLS_static(std::shared_ptr&lt;GraphTask&gt;, tls_current_graph_task);">`shared_ptr`</SwmToken> reference to <SwmToken path="test/dynamo/test_global.py" pos="185:3:3" line-data="        class Variable:">`Variable`</SwmToken> and references to other Python objects needed by the Python runtime.

<SwmSnippet path="/torch/csrc/autograd/engine.cpp" line="1190">

---

The <SwmToken path="torch/csrc/autograd/engine.cpp" pos="1190:4:4" line-data="auto Engine::execute(">`execute`</SwmToken> function in the Autograd engine is responsible for executing the backward pass, computing gradients for all Tensors that require them. This function validates outputs and handles potential memory leaks when using <SwmToken path="torch/csrc/autograd/engine.cpp" pos="1204:4:6" line-data="        &quot;Using backward() with create_graph=True will create a reference cycle &quot;">`backward()`</SwmToken> with <SwmToken path="torch/csrc/autograd/engine.cpp" pos="1204:10:12" line-data="        &quot;Using backward() with create_graph=True will create a reference cycle &quot;">`create_graph=True`</SwmToken>.

```c++
auto Engine::execute(
    const edge_list& root_edges,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) -> variable_list {
  validate_outputs(
      root_edges,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<variable_list&>(inputs),
      [](const std::string& msg) { return msg; });
  if (accumulate_grad && create_graph) {
    TORCH_WARN_ONCE(
        "Using backward() with create_graph=True will create a reference cycle "
        "between the parameter and its gradient which can cause a memory leak. "
        "We recommend using autograd.grad when creating the graph to avoid this. "
        "If you have to use this function, make sure to reset the .grad fields of "
        "your parameters to None after use to break the cycle and avoid the leak.");
  }
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/autograd/autograd.cpp" line="140">

---

The <SwmToken path="benchmarks/operator_benchmark/benchmark_pytorch.py" pos="174:3:3" line-data="    def run_backward(self, num_runs, print_per_iter=False):">`run_backward`</SwmToken> function calls <SwmToken path="torch/csrc/autograd/engine.cpp" pos="1190:2:4" line-data="auto Engine::execute(">`Engine::execute`</SwmToken> to perform the backward pass and compute gradients.

```c++
  variable_list grad_inputs = Engine::get_default_engine().execute(
      roots,
```

---

</SwmSnippet>

# Forward-mode Automatic Differentiation

Forward-mode automatic differentiation is one of the methods provided by Autograd to compute derivatives efficiently.

# Locally Disabling Gradient Computation

You can locally disable gradient computation using context managers like `torch.no_grad()` to improve performance during inference.

# Custom Function Utilities

Autograd allows you to define custom autograd functions by subclassing <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1163:16:20" line-data="      &quot;https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function &quot;);">`torch.autograd.Function`</SwmToken> and implementing the <SwmToken path="benchmarks/instruction_counts/core/api.py" pos="36:1:1" line-data="    FORWARD = &quot;Forward&quot;">`FORWARD`</SwmToken> and <SwmToken path="torch/csrc/autograd/engine.cpp" pos="1204:4:4" line-data="        &quot;Using backward() with create_graph=True will create a reference cycle &quot;">`backward`</SwmToken> static methods.

# Autograd Endpoints

User <SwmToken path="test/functorch/discover_coverage.py" pos="91:1:1" line-data="    apis = get_public_overridable_apis()">`apis`</SwmToken> provide various properties and methods for the <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="76:1:1" line-data="    THPFunction* self,">`THPFunction`</SwmToken> class.

<SwmSnippet path="/torch/csrc/autograd/python_function.cpp" line="1697">

---

The <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1697:6:6" line-data="static struct PyGetSetDef THPFunction_properties[] = {">`THPFunction_properties`</SwmToken> structure defines various properties for the <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="76:1:1" line-data="    THPFunction* self,">`THPFunction`</SwmToken> class, including <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1698:3:3" line-data="    {&quot;saved_tensors&quot;,">`saved_tensors`</SwmToken>, <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1703:3:3" line-data="    {&quot;saved_variables&quot;,">`saved_variables`</SwmToken>, and <SwmToken path="torch/csrc/autograd/python_function.cpp" pos="1713:3:3" line-data="    {&quot;next_functions&quot;,">`next_functions`</SwmToken>.

```c++
static struct PyGetSetDef THPFunction_properties[] = {
    {"saved_tensors",
     (getter)THPFunction_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"saved_variables",
     (getter)THPFunction_saved_variables,
     nullptr,
     nullptr,
     nullptr},
    {"_raw_saved_tensors",
     (getter)THPFunction_raw_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"next_functions",
     (getter)THPFunction_next_functions,
     nullptr,
     nullptr,
     nullptr},
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
