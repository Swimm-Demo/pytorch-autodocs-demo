---
title: Debugging the Compilation Process
---
This document will cover the process of debugging the compilation process of an FX graph, which includes:

1. Saving the FX graph
2. Printing instructions for minimizing the FX graph
3. Compiling the FX graph.

Technical document: <SwmLink doc-title="Debugging the Compilation Process">[Debugging the Compilation Process](/.swm/debugging-the-compilation-process.9uqnk6a0.sw.md)</SwmLink>

# [Saving the FX Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/9uqnk6a0#the-debug_compile-function)

The first step in debugging the compilation process is to save the FX graph to a folder. This allows for a persistent record of the graph's state at this point in the process. By saving the graph, users can later retrieve and inspect it to understand its structure and any potential issues.

# [Printing Instructions for Minimizing the FX Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/9uqnk6a0#the-debug_compile-function)

After saving the FX graph, the next step is to print out instructions that guide the user on how to minimize the FX graph. These instructions are crucial as they provide a step-by-step guide on reducing the graph to its simplest form, making it easier to identify and isolate issues. The instructions typically include commands and scripts that the user can run to perform the minimization.

# [Compiling the FX Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/9uqnk6a0#the-ts_compile-function)

The final step in the debugging process is to compile the FX graph using the `ts_compile` function. This function takes the FX graph and performs several transformations to optimize it for execution. These transformations include stripping overloads, replacing certain operations, scripting the graph, removing mutations, and freezing the scripted model. The result is an optimized TorchScript model that is ready for inference. This step ensures that the FX graph is not only free of errors but also optimized for performance.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
