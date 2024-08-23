---
title: Verifying Optimizer State Dictionary
---
This document will cover the process of verifying the optimizer state dictionary (OSD). We'll cover:

1. Gathering the state dictionary
2. Setting the state dictionary for the model and new optimizer
3. Comparing the state dictionaries of the original and new optimizers.

Technical document: <SwmLink doc-title="Verifying Optimizer State Dictionary">[Verifying Optimizer State Dictionary](/.swm/verifying-optimizer-state-dictionary.wmq2d5zv.sw.md)</SwmLink>

# [Gathering the State Dictionary](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/wmq2d5zv#_gather_state_dict)

The process begins by gathering the state dictionary. This involves collecting all the necessary data from the distributed components. The gathered state dictionary consolidates the state from various parts of the system, ensuring that it is complete and ready for verification.

# [Setting the State Dictionary for the Model and New Optimizer](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/wmq2d5zv#set_state_dict)

Next, the gathered state dictionary is set for both the model and a new optimizer. This step ensures that the model and the new optimizer are initialized with the same state as the original optimizer. This is crucial for maintaining consistency and ensuring that the new optimizer can be compared accurately with the original.

# [Comparing the State Dictionaries](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/wmq2d5zv#_verify_osd_by_load)

Finally, the state dictionaries of the original and new optimizers are compared. This comparison checks for any discrepancies between the two state dictionaries. If they are equal, it confirms that the state has been correctly loaded and maintained. This step is essential for verifying the integrity of the optimizer state across different training sessions.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
