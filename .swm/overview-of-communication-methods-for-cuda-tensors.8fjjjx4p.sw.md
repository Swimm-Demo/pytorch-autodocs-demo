---
title: Overview of Communication Methods for CUDA Tensors
---
This document will cover the setup of communication methods for CUDA tensors, which includes:

1. Broadcasting tensors
2. Scattering tensors
3. Gathering tensors.

Technical document: <SwmLink doc-title="Overview of initCommMethods">[Overview of initCommMethods](/.swm/overview-of-initcommmethods.eesf23mt.sw.md)</SwmLink>

# [Broadcasting Tensors](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eesf23mt#broadcast_out)

Broadcasting tensors involves sending a single tensor to multiple devices. This ensures that each device has an identical copy of the tensor. The process checks that all output tensors are CUDA tensors and have the same shape as the source tensor. This is crucial for synchronizing tensor data across multiple devices, ensuring consistency and efficiency in data handling.

# [Scattering Tensors](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eesf23mt#scatter_out)

Scattering tensors involves splitting a source tensor into smaller chunks and distributing these chunks across multiple devices. Each chunk is sent to a different device. The process ensures that the output tensors are CUDA tensors and have the correct shape. This method is useful for parallel processing, where different parts of the tensor can be processed simultaneously on different devices.

# [Gathering Tensors](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eesf23mt#gather_out)

Gathering tensors involves collecting multiple tensors from different devices into a single output tensor. The process ensures that all input tensors are CUDA tensors and have the same number of dimensions. This method is essential for consolidating data processed on different devices into a single tensor for further operations or analysis.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
