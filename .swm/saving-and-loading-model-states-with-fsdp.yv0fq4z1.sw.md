---
title: Saving and Loading Model States with FSDP
---
This document will cover the process of saving and loading model states using Fully Sharded Data Parallel (FSDP). We'll cover:

1. Setting up the environment
2. Saving the model state
3. Loading the model state
4. Transforming the optimizer state dictionary

Technical document: <SwmLink doc-title="Saving and Loading Model States with FSDP">[Saving and Loading Model States with FSDP](/.swm/saving-and-loading-model-states-with-fsdp.eo5dypeg.sw.md)</SwmLink>

# [Setting up the environment](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eo5dypeg#saving-and-loading-model-states)

The process begins by setting up the environment. This involves configuring the necessary settings for distributed training. The environment setup includes specifying the master address and port, initializing the process group, and setting the device for each rank. This step ensures that all processes can communicate effectively and are ready for distributed operations.

# [Saving the model state](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eo5dypeg#saving-the-model-state)

Once the environment is set up, the next step is to save the model state. This involves creating a model and an optimizer, and then saving their states to a checkpoint. The model state and optimizer state are saved using a specific state dictionary type called `SHARDED_STATE_DICT`. This type ensures that the state is saved in a way that is compatible with sharded models. The state dictionary is then saved to a specified directory, which allows it to be loaded later.

# [Loading the model state](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eo5dypeg#loading-the-model-state)

After saving the model state, the next step is to load it into a new model. This ensures that the new model has the same parameters as the original model. The loading process involves reading the saved state dictionary from the specified directory and loading it into the new model. The optimizer state is loaded separately to ensure that it is correctly mapped to the new model. This step is crucial for resuming training or evaluation from a saved checkpoint.

# [Transforming the optimizer state dictionary](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/eo5dypeg#transforming-optimizer-state-dictionary)

The final step involves transforming the optimizer state dictionary. This transformation is necessary to ensure that the optimizer state is compatible with the sharded model. The optimizer state dictionary can be transformed into different types, such as a full optimizer state dictionary, a sharded optimizer state dictionary, or a local optimizer state dictionary. This flexibility allows the optimizer state to be used in various scenarios, depending on the specific requirements of the training or evaluation process.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
