---
title: Getting Started with PyTorch Modules
---
# Getting Started with Modules in Nn

Modules are used to represent neural networks. They serve as the building blocks of stateful computation. Modules are tightly integrated with PyTorch's autograd system, making it simple to specify learnable parameters for optimizers to update. They are easy to work with and transform, allowing for straightforward saving, restoring, and transferring between devices. Modules can be pruned, quantized, and more, providing flexibility in neural network construction and optimization.

## Modules as Building Blocks

Modules are the building blocks of stateful computation. <SwmToken path="tools/nightly.py" pos="306:1:1" line-data="    pytorch, platform = &quot;&quot;, &quot;&quot;">`pytorch`</SwmToken> provides a robust library of modules and makes it simple to define new custom modules, allowing for easy construction of elaborate, multi-layer neural networks.

## Integration with Autograd

Modules are tightly integrated with PyTorch's autograd system. This integration makes it simple to specify learnable parameters for PyTorch's optimizers to update.

## Ease of Use and Transformation

Modules are straightforward to save and restore, transfer between CPU/GPU/TPU devices, prune, quantize, and more. This makes them highly flexible for various neural network construction and optimization tasks.

## Module State

A module's <SwmToken path="torch/nn/modules/module.py" pos="2090:14:14" line-data="        submodule in :meth:`~torch.nn.Module.state_dict`.">`state_dict`</SwmToken> contains state that affects its computation, including parameters and buffers. Modules can have state beyond parameters that affects computation but is not learnable, such as persistent and <SwmToken path="torch/nn/modules/module.py" pos="529:17:19" line-data="        only difference between a persistent buffer and a non-persistent buffer">`non-persistent`</SwmToken> buffers.

# Main Functions

There are several main functions in modules. Some of them are <SwmToken path="torch/nn/modules/module.py" pos="416:3:3" line-data="            def forward(self, x):">`forward`</SwmToken>, <SwmToken path="torch/nn/modules/module.py" pos="2090:14:14" line-data="        submodule in :meth:`~torch.nn.Module.state_dict`.">`state_dict`</SwmToken>, and <SwmToken path="test/distributed/checkpoint/e2e/test_e2e_save_and_load.py" pos="78:3:3" line-data="    def load_state_dict(self, state_dict):">`load_state_dict`</SwmToken>. We will dive a little into <SwmToken path="torch/nn/modules/module.py" pos="416:3:3" line-data="            def forward(self, x):">`forward`</SwmToken> and <SwmToken path="torch/nn/modules/module.py" pos="2090:14:14" line-data="        submodule in :meth:`~torch.nn.Module.state_dict`.">`state_dict`</SwmToken>.

<SwmSnippet path="/torch/nn/modules/module.py" line="416">

---

## forward

The <SwmToken path="torch/nn/modules/module.py" pos="416:3:3" line-data="            def forward(self, x):">`forward`</SwmToken> function defines the computation performed at every call. It should be overridden by all subclasses.

```python
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="2085">

---

## <SwmToken path="torch/nn/modules/module.py" pos="2090:14:14" line-data="        submodule in :meth:`~torch.nn.Module.state_dict`.">`state_dict`</SwmToken>

The <SwmToken path="torch/nn/modules/module.py" pos="2090:14:14" line-data="        submodule in :meth:`~torch.nn.Module.state_dict`.">`state_dict`</SwmToken> function returns a dictionary containing a whole state of the module, including parameters and persistent buffers.

```python
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="2101">

---

## <SwmToken path="test/distributed/checkpoint/e2e/test_e2e_save_and_load.py" pos="78:3:3" line-data="    def load_state_dict(self, state_dict):">`load_state_dict`</SwmToken>

The <SwmToken path="test/distributed/checkpoint/e2e/test_e2e_save_and_load.py" pos="78:3:3" line-data="    def load_state_dict(self, state_dict):">`load_state_dict`</SwmToken> function loads a module's state from a dictionary. This is useful for restoring a model's state.

```python
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", Module.get_extra_state)
            is not Module.get_extra_state
        ):
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
