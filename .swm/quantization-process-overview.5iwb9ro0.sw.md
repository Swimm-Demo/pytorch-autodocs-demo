---
title: Quantization Process Overview
---
This document will cover the Quantization Process Overview, which includes:

1. Initializing the model and setting up the quantizer
2. Preparing the model for quantization-aware training or post-training quantization
3. Converting the model into a quantized version for efficient inference.

Technical document: <SwmLink doc-title="Quantization Process Overview">[Quantization Process Overview](/.swm/quantization-process-overview.10rutu09.sw.md)</SwmLink>

# [Initializing the Model and Setting Up the Quantizer](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/10rutu09#_get_pt2e_quantized_linear)

The first step in the quantization process involves initializing a simple linear model and setting up a quantizer. The quantizer is configured with a symmetric quantization configuration. This setup is crucial as it defines how the model's weights and activations will be quantized, ensuring that the model can be efficiently converted later.

# [Preparing the Model for Quantization](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/10rutu09#_quantize)

Once the model is initialized, it needs to be prepared for quantization. This preparation can be done in two ways: quantization-aware training (QAT) or post-training quantization (PTQ). QAT involves training the model with quantization in mind, adding fake quantization modules to simulate the effects of quantization during training. PTQ, on the other hand, involves quantizing the model after it has been trained. The choice between QAT and PTQ depends on the specific requirements and constraints of the deployment environment.

# [Converting the Model to a Quantized Version](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/10rutu09#convert_pt2e)

After the model is prepared, it is converted into a quantized version. This conversion process involves folding the quantize operations and rewriting the model to use a reference representation if specified. The goal is to optimize the model for efficient inference on quantized hardware, ensuring that it can perform computations more efficiently while maintaining accuracy.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
