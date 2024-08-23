---
title: Overview of aot_dispatch_autograd
---
This document will cover the overview of the `aot_dispatch_autograd` feature, which includes:

1. Generating a joint graph
2. Partitioning the graph
3. Manipulating the input with various wrappers
4. Returning a wrapped `torch.autograd.Function`

Technical document: <SwmLink doc-title="Overview of aot_dispatch_autograd">[Overview of aot_dispatch_autograd](/.swm/overview-of-aot_dispatch_autograd.tfnc9hc2.sw.md)</SwmLink>

# [Generating a Joint Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/tfnc9hc2#aot_dispatch_autograd_graph)

The process begins by generating a joint graph. This involves creating a combined representation of the forward and backward passes of a function. The joint graph is essential for tracing and functionalizing the function, ensuring that all necessary inputs and metadata are correctly managed. This step sets up the foundation for further processing and optimization.

# [Partitioning the Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/tfnc9hc2#partition_fn)

Once the joint graph is generated, it is partitioned into separate forward and backward graphs. This partitioning is done to optimize memory and computation trade-offs. By recomputing forward operations during the backward pass, the system can reduce memory usage. This step involves dead code elimination, node classification, and using a memory budget to determine which values to save for the backward pass.

# [Manipulating the Input with Various Wrappers](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/tfnc9hc2#aot_dispatch_autograd)

After partitioning the graph, the input is manipulated with various wrappers. These wrappers are created to handle the dispatch logic and ensure that the function and its arguments are pre-compiled with deterministic metadata. This step is crucial for preparing the function for further processing and ensuring that the autograd logic is correctly handled.

# [Returning a Wrapped torch.autograd.Function](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/tfnc9hc2#aot_dispatch_autograd)

Finally, the function returns a wrapped `torch.autograd.Function` that includes both the forward and backward passes. This wrapped function is now ready for further processing and can be used in various autograd operations. This step ensures that the function is fully prepared for execution with all necessary optimizations and configurations in place.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
