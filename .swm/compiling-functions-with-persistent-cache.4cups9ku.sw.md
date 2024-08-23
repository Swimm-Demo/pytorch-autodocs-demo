---
title: Compiling Functions with Persistent Cache
---
This document will cover the process of compiling a function with a persistent cache, which includes:

1. Ensuring static shapes and validating input types
2. Creating necessary directories and setting environment variables
3. Compiling the function and generating metadata

Technical document: <SwmLink doc-title="Compiling with Persistent Cache">[Compiling with Persistent Cache](/.swm/compiling-with-persistent-cache.nsnu9ga8.sw.md)</SwmLink>

# [Ensuring static shapes and validating input types](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/nsnu9ga8#ensuring-static-shapes-and-validating-input-types)

The process begins by ensuring that only static shapes are supported. This means that the shapes of the inputs to the function must not change. This is important because it allows the compiled function to be reused without needing to be recompiled for different input shapes. Additionally, the input types are validated to ensure they are compatible with the function being compiled.

# [Creating necessary directories and setting environment variables](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/nsnu9ga8#creating-necessary-directories-and-setting-environment-variables)

Next, the necessary directories for the persistent cache are created. This involves setting up a specific directory structure where the compiled function and its metadata will be stored. An environment variable, `TORCHINDUCTOR_CACHE_DIR`, is set to point to this cache directory. This ensures that the compiled function and its metadata can be easily located and accessed for future use.

# [Compiling the function and generating metadata](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/nsnu9ga8#compiling-the-function-and-generating-metadata)

The function is then compiled using the provided inputs. During this process, metadata about the inputs is generated. This metadata includes information about the input types and shapes, which is necessary for ensuring that the compiled function can be reused with the same inputs. The metadata is stored in a JSON file within the persistent cache directory. This allows the function to be efficiently reused without needing to be recompiled, saving time and computational resources.

# [Storing metadata in persistent cache directory](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==/docs/nsnu9ga8#storing-metadata-in-persistent-cache-directory)

Finally, the generated metadata is stored in the persistent cache directory. This ensures that the compiled function and its associated metadata are kept together in a single location. By storing the metadata in a JSON file, it can be easily accessed and read by other processes or functions that need to use the compiled function. This step is crucial for enabling the efficient reuse of the compiled function without needing to go through the entire compilation process again.

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
