---
title: Generating a Quantized Model
---
This document will cover the process of generating a quantized model, which includes:

1. Capturing the pre-autograd graph
2. Preparing the model for Quantization Aware Training (QAT)
3. Converting the model to a quantized version.

Technical document: <SwmLink doc-title="Generating a Quantized Model">[Generating a Quantized Model](/.swm/generating-a-quantized-model.ami9p9e2.sw.md)</SwmLink>

# [Capturing the Pre-Autograd Graph](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/ami9p9e2#capture_pre_autograd_graph)

The first step in generating a quantized model is to capture the computational graph of the model before any automatic differentiation (autograd) occurs. This is important because it preserves the original structure and operations of the model, which is necessary for accurate quantization. By capturing the graph at this stage, we ensure that the quantization process does not interfere with the model's training dynamics.

# [Preparing the Model for Quantization Aware Training (QAT)](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/ami9p9e2#prepare_qat_pt2e)

If Quantization Aware Training (QAT) is required, the model needs to be prepared accordingly. This involves annotating the model with quantization information and performing necessary fusions, such as combining convolution and batch normalization layers. These steps are crucial for optimizing the model for quantization, ensuring that it can be trained with quantization effects in mind. The prepared model is then ready for QAT, which allows it to learn how to handle quantized operations during training.

# [Converting the Model to a Quantized Version](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/ami9p9e2#convert_pt2e)

Once the model is prepared, it is moved to evaluation mode and converted into a quantized model. This conversion process involves several steps, including folding convolution and batch normalization layers to reduce computational complexity. Additionally, the model may be rewritten to use a reference representation, which optimizes the graph for quantized operations. The final result is a model that is ready for deployment with quantized weights and operations, offering improved performance and reduced resource usage.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
