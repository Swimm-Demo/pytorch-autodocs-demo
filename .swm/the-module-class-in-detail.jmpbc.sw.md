---
title: The Module class in Detail
---
This document will cover the `Module` class in `torch/nn/modules/module.py`. We will cover:

1. What `Module` is.
2. Variables and functions in `Module`.

# What is Module

The `Module` class in `torch/nn/modules/module.py` is the base class for all neural network modules in PyTorch. It provides a way to define and manage the parameters, buffers, and submodules of a neural network. Modules can be nested within each other, allowing for complex model architectures. The `Module` class also provides methods for moving the module to different devices, applying functions to its parameters, and saving/loading its state.

<SwmSnippet path="/torch/nn/modules/module.py" line="432">

---

# Variables and functions

The variable `dump_patches` is a class attribute that is set to `False` by default. It is used for backward compatibility support.

```python
    dump_patches: bool = False
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="434">

---

The variable `_version` is an integer that allows better backward compatibility support for the `load_state_dict` method. It is used to track changes in the module's parameters and buffers.

```python
    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="447">

---

The `__init__` function initializes the internal state of the `Module`. It sets up various dictionaries to store parameters, buffers, hooks, and submodules. It also sets the module to training mode by default.

```python
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_pre_hooks: Dict[int, Callable]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: Dict[int, bool]
    # forward hooks that should always be called even if an exception is raised
    _forward_hooks_always_called: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="519">

---

The `register_buffer` function adds a buffer to the module. Buffers are tensors that are not considered parameters but are part of the module's state. This function allows specifying whether the buffer should be persistent.

```python
    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="581">

---

The `register_parameter` function adds a parameter to the module. Parameters are tensors that are considered part of the module's learnable state. This function allows specifying whether the parameter should be included in the module's `state_dict`.

```python
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(
                f"parameter name should be a string. Got {torch.typename(name)}"
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="631">

---

The `add_module` function adds a child module to the current module. Child modules can be accessed as attributes using the given name. This function ensures that the child module is properly registered and its parameters are converted when calling methods like `to`.

```python
    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(
                f"module name should be a string. Got {torch.typename(name)}"
            )
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="659">

---

The `register_module` function is an alias for the `add_module` function.

```python
    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="664">

---

The `get_submodule` function returns the submodule given by the target string if it exists. It allows checking for the existence of nested submodules and is more efficient than querying `named_modules`.

```python
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="728">

---

The `set_submodule` function sets the submodule given by the target string if it exists. It allows overriding existing submodules with new ones.

```python
    def set_submodule(self, target: str, module: "Module") -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="787">

---

The `get_parameter` function returns the parameter given by the target string if it exists. It allows checking for the existence of nested parameters.

```python
    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="823">

---

The `get_buffer` function returns the buffer given by the target string if it exists. It allows checking for the existence of nested buffers.

```python
    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="859">

---

The `get_extra_state` function is intended to return any extra state to include in the module's `state_dict`. It should be implemented by subclasses if they need to store extra state.

```python
    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="880">

---

The `set_extra_state` function is intended to set extra state contained in the loaded `state_dict`. It should be implemented by subclasses if they need to load extra state.

```python
    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="897">

---

The `_apply` function applies a given function to all parameters and buffers of the module. It is used for operations like moving the module to a different device or changing the data type of its parameters.

```python
    def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        should_use_swap_tensors = (
            torch.__future__.get_swap_module_params_on_conversion()
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="992">

---

The `apply` function applies a given function recursively to every submodule as well as the module itself. It is typically used for initializing the parameters of a model.

```python
    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1033">

---

The `cuda` function moves all model parameters and buffers to the GPU. It modifies the module in-place and should be called before constructing the optimizer if the module will live on the GPU while being optimized.

```python
    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1052">

---

The `ipu` function moves all model parameters and buffers to the IPU. It modifies the module in-place and should be called before constructing the optimizer if the module will live on the IPU while being optimized.

```python
    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.ipu(device))
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1071">

---

The `xpu` function moves all model parameters and buffers to the XPU. It modifies the module in-place and should be called before constructing the optimizer if the module will live on the XPU while being optimized.

```python
    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.xpu(device))
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1090">

---

The `mtia` function moves all model parameters and buffers to the MTIA. It modifies the module in-place and should be called before constructing the optimizer if the module will live on the MTIA while being optimized.

```python
    def mtia(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the MTIA.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on MTIA while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.mtia(device))
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1109">

---

The `cpu` function moves all model parameters and buffers to the CPU. It modifies the module in-place.

```python
    def cpu(self: T) -> T:
        r"""Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1120">

---

The `type` function casts all parameters and buffers to the specified data type. It modifies the module in-place.

```python
    def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1134">

---

The `float` function casts all floating point parameters and buffers to the `float` data type. It modifies the module in-place.

```python
    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1145">

---

The `double` function casts all floating point parameters and buffers to the `double` data type. It modifies the module in-place.

```python
    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1156">

---

The `half` function casts all floating point parameters and buffers to the `half` data type. It modifies the module in-place.

```python
    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/nn/modules/module.py" line="1167">

---

The `bfloat16` function casts all floating point parameters and buffers to the `bfloat16` data type. It modifies the module in-place.

```python
    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
