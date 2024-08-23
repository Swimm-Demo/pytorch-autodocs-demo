---
title: Converting a Model to NNAPI
---
This document will cover the process of converting a PyTorch model to NNAPI format, which includes:

1. Preparing the model for NNAPI
2. Creating necessary modules
3. Scripting the model for execution on NNAPI.

Technical document: <SwmLink doc-title="Converting a Model to NNAPI">[Converting a Model to NNAPI](/.swm/converting-a-model-to-nnapi.0zt4zne6.sw.md)</SwmLink>

# [Preparing the Model for NNAPI](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/0zt4zne6#processing-the-model-for-nnapi)

The first step in converting a PyTorch model to NNAPI is to prepare the model. This involves computing the shapes of the tensors, serializing the model tensor, and gathering the necessary weights and memory formats. This preparation ensures that the model is in a format that can be processed by NNAPI. The serialized model includes information such as used weights, input and output memory formats, and the return value count.

# [Creating Necessary Modules](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/0zt4zne6#creating-the-shape-computation-module)

After preparing the model, the next step is to create the necessary modules. A specialized module called `ShapeComputeModule` is created to handle tensor shape computation. This module is scripted to dynamically add a method that will mutate the serialized model according to the computed operand shapes based on the input arguments. This step ensures that the model can adapt to different input shapes during execution.

# [Scripting the Model for Execution on NNAPI](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/0zt4zne6#scripting-the-model)

The final step in the conversion process is scripting the model. This involves converting the PyTorch model into TorchScript, a statically-typed subset of Python that can be optimized and run independently of Python. The scripting process inspects the source code of the model, compiles it using the TorchScript compiler, and returns a `ScriptModule` or `ScriptFunction`. This conversion is crucial for enabling the model to run efficiently on different backends, including NNAPI.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
