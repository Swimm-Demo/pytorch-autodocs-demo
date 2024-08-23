---
title: Overview of simple_ts_compile
---
This document provides an overview of the `simple_ts_compile` function, which is responsible for compiling and freezing a TorchScript function. The process involves several key steps, including stripping overloads from the graph, scripting the graph into a TorchScript function, and freezing the resulting function to optimize it for execution.

The `simple_ts_compile` function starts by removing any unnecessary overloads from the graph. It then converts the graph into a TorchScript function, which is a more efficient representation for execution. Finally, the function freezes the TorchScript function, which involves optimizing the graph by converting certain parameters into constants and applying various optimization passes. This makes the function more efficient and ready for execution.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      1e675e062b7382add841df57d18ec25733a1e66bfa887c1cc03d71c4152595b9(simple_ts_compile):::mainFlowStyle --> 68ce5dd4f3c4b98eabc8c8e262beceb25f798a453e583233f9c67a3831473ed3(script)

subgraph torchinductor["torch/_inductor"]
1e675e062b7382add841df57d18ec25733a1e66bfa887c1cc03d71c4152595b9(simple_ts_compile):::mainFlowStyle --> 25326b33f3cc300e77bfb76058518d86fdf74955e966fd4a354c729b5fdad3c8(freeze):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
25326b33f3cc300e77bfb76058518d86fdf74955e966fd4a354c729b5fdad3c8(freeze):::mainFlowStyle --> 93d6ef92ae74c60fde33308be0601add41819a1c5e68dc719c8cd01720ea5508(freezing_passes):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
93d6ef92ae74c60fde33308be0601add41819a1c5e68dc719c8cd01720ea5508(freezing_passes):::mainFlowStyle --> 886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle --> 19b7c6e3ef4d2e87b2c4187afffdb8068b30e6b0eaa86c56b55a1f0acc28e6c0(_mkldnn_weight_pack_init)
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle --> e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle --> 8b759173446895518816ce40520945a0d5c683c340ecd04383824890173a351d(addmm)
end

subgraph torchinductor["torch/_inductor"]
e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle --> 5dd00346c8f29cefda91fc3dbe3db612537d60682f6450f5df325f8b0cbede3b(register_replacement):::mainFlowStyle
end

subgraph torchinductor["torch/_inductor"]
5dd00346c8f29cefda91fc3dbe3db612537d60682f6450f5df325f8b0cbede3b(register_replacement):::mainFlowStyle --> fb2a894ced1d21f5513b07111a77e9a3b1457976bbb1246c3f1e68dbba73009f(gen_pattern):::mainFlowStyle
end

subgraph torchfunctorch["torch/_functorch"]
fb2a894ced1d21f5513b07111a77e9a3b1457976bbb1246c3f1e68dbba73009f(gen_pattern):::mainFlowStyle --> 3192db0a19b37cdee5ce1d03ca49edd9406e9385950b7b77164328a7e0b5f559(trace_fn):::mainFlowStyle
end

3192db0a19b37cdee5ce1d03ca49edd9406e9385950b7b77164328a7e0b5f559(trace_fn):::mainFlowStyle --> 46d0c3ff106c08d7937702fd73ddbb320c4c519be87b38dcedc625835f91b8a8(make_dual):::mainFlowStyle

46d0c3ff106c08d7937702fd73ddbb320c4c519be87b38dcedc625835f91b8a8(make_dual):::mainFlowStyle --> de7c515ae4b8c45003b6984425b592ce24c09e8a3d9d2022e0747b452ffd3b39(_make_dual):::mainFlowStyle


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

# Flow drill down

First, we'll zoom into this section of the flow:

```mermaid
graph TD;
      1e675e062b7382add841df57d18ec25733a1e66bfa887c1cc03d71c4152595b9(simple_ts_compile):::mainFlowStyle --> 68ce5dd4f3c4b98eabc8c8e262beceb25f798a453e583233f9c67a3831473ed3(script)

1e675e062b7382add841df57d18ec25733a1e66bfa887c1cc03d71c4152595b9(simple_ts_compile):::mainFlowStyle --> 25326b33f3cc300e77bfb76058518d86fdf74955e966fd4a354c729b5fdad3c8(freeze):::mainFlowStyle

25326b33f3cc300e77bfb76058518d86fdf74955e966fd4a354c729b5fdad3c8(freeze):::mainFlowStyle --> ml0x8(...)

68ce5dd4f3c4b98eabc8c8e262beceb25f798a453e583233f9c67a3831473ed3(script) --> 501552a94ffa434d657921341f525cbfba4d9c2d944698152f533359772d9a45(_script_impl)


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_functorch/compilers.py" line="191">

---

## Compiling and Freezing a TorchScript Function

The function `simple_ts_compile` first strips overloads from the given graph using `strip_overloads(fx_g)`. It then compiles the graph into a TorchScript function using `torch.jit.script(fx_g)` and freezes the resulting function with `torch.jit.freeze(f.eval())`.

```python
    strip_overloads(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
```

---

</SwmSnippet>

<SwmSnippet path="/torch/jit/_script.py" line="1224">

---

## Scripting a Function

The `script` function is responsible for converting a Python function, module, dictionary, or list into a TorchScript representation. This is done by inspecting the source code and compiling it using the TorchScript compiler.

```python
    r"""Script the function.

    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be
    subsequently passed by reference between Python and TorchScript with zero copy overhead.

    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists
     and as a decorator ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.
```

---

</SwmSnippet>

<SwmSnippet path="/torch/jit/_script.py" line="1255">

---

### Example of Scripting a Function

An example of scripting a function is provided, where a simple function `foo` is decorated with `@torch.jit.script`. This converts the function into a `ScriptFunction`, which can then be executed using the TorchScript interpreter.

```python
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(foo.code)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/jit/_script.py" line="1086">

---

## Implementation of Script Function

The `_script_impl` function handles the actual conversion of the given object into a TorchScript representation. It checks the type of the object and processes it accordingly, whether it's a module, function, class, dictionary, or list.

```python
def _script_impl(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    global type_trace_db

    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=3,
        )

    # No-op for modules, functions, class instances that are already scripted
    if isinstance(obj, RecursiveScriptClass):
        return obj
    if isinstance(obj, ScriptModule):
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
25326b33f3cc300e77bfb76058518d86fdf74955e966fd4a354c729b5fdad3c8(freeze):::mainFlowStyle --> 93d6ef92ae74c60fde33308be0601add41819a1c5e68dc719c8cd01720ea5508(freezing_passes):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
93d6ef92ae74c60fde33308be0601add41819a1c5e68dc719c8cd01720ea5508(freezing_passes):::mainFlowStyle --> 886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle --> 19b7c6e3ef4d2e87b2c4187afffdb8068b30e6b0eaa86c56b55a1f0acc28e6c0(_mkldnn_weight_pack_init)
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
886db690b8db4efc69c2920e92e12999717b8d59945f19216af42d4c4de6d858(lazy_init):::mainFlowStyle --> e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle --> zpu1j(...)
end

subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
19b7c6e3ef4d2e87b2c4187afffdb8068b30e6b0eaa86c56b55a1f0acc28e6c0(_mkldnn_weight_pack_init) --> cbcd26e7cf9042d804fc4a3d71f501531e6ada6a93ae89ef7577a5218c0d0d47(_register_quantization_weight_pack_pass)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_inductor/freezing.py" line="66">

---

## Freeze

The `freeze` function inlines parameters that are not mutated into constants and optimizes the graph through constant propagation and other techniques. It also discards the original parameters of the module for memory efficiency if enabled. This function is run in dynamo tracing post aot_autograd.

```python
def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    # We have convert conv's weight to channels last which may meet error for .view
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/freezing_patterns.py" line="36">

---

## Freezing Passes

The `freezing_passes` function applies various passes to the graph to freeze it. This includes constant folding, binary folding, and applying specific patterns to optimize the graph further.

```python
def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """

    from ..freezing import constant_fold

    lazy_init()
    # We need a few rounds of binary folding to get rid of all the
    # unnecessary nodes, but may need a good method to chose the rounds number.
    # works like: conv+binary+binary.
    binary_folding = counters["inductor"]["binary_folding"]
    fake_tensor_prop(gm, aot_example_inputs, True)

    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_convs(gm)
    for _ in range(4):
        constant_fold(gm)
        # Make sure meta['val'] is properly set for all nodes
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)  # type: ignore[arg-type]
        # If we don't have binary folding, we don't need to run the pass again.
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/freezing_patterns.py" line="87">

---

## Lazy Initialization

The `lazy_init` function initializes various components required for freezing, such as MKL-DNN weight packing and binary folding.

```python
def lazy_init():
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    from .binary_folding import binary_folding_init

    addmm_patterns_init()
    binary_folding_init()
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/mkldnn_fusion.py" line="1262">

---

## MKL-DNN Weight Pack Initialization

The `_mkldnn_weight_pack_init` function registers the weight pack pass and recovers linear operations if MKL-DNN is enabled and available.

```python
    def _mkldnn_weight_pack_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()
            _recover_linear()
            _register_quantization_weight_pack_pass()
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/quantization.py" line="2490">

---

## Register Quantization Weight Pack Pass

The `_register_quantization_weight_pack_pass` function registers various steps for quantization weight packing, including dequant promotion and QConv/QLinear weight prepack.

```python
def _register_quantization_weight_pack_pass():
    # Step 1: Dequant promotion for int8-mixed-fp32/bf16
    _register_dequant_promotion()

    # Step 2: QConv weight prepack
    _register_qconv_weight_prepack()

    # Step 3: QLinear weight prepack
    _register_qlinear_weight_prepack()
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchinductorfxpasses["torch/_inductor/fx_passes"]
e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle --> 8b759173446895518816ce40520945a0d5c683c340ecd04383824890173a351d(addmm)
end

subgraph torchinductorpatternmatcherpy["torch/_inductor/pattern_matcher.py"]
e27ba5abddcd6d34a6d5346aa27bc69a70318fd7ade09d0e45ea3fe93b619a94(addmm_patterns_init):::mainFlowStyle --> 5dd00346c8f29cefda91fc3dbe3db612537d60682f6450f5df325f8b0cbede3b(register_replacement):::mainFlowStyle
end

subgraph torchinductorpatternmatcherpy["torch/_inductor/pattern_matcher.py"]
5dd00346c8f29cefda91fc3dbe3db612537d60682f6450f5df325f8b0cbede3b(register_replacement):::mainFlowStyle --> 9kmag(...)
end

subgraph torchinductorpatternmatcherpy["torch/_inductor/pattern_matcher.py"]
8b759173446895518816ce40520945a0d5c683c340ecd04383824890173a351d(addmm) --> 2a2971e4ef0e5aef2654828668c70787a1e8e6fbdd0951fe915dc219096f441f(replace_by_example)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_inductor/fx_passes/freezing_patterns.py" line="116">

---

## Initializing addmm Patterns

The function `addmm_patterns_init` initializes patterns for matrix multiplication and addition operations. It first determines the device (CPU or CUDA) and sets up a partial function to create empty tensors. The function defines several patterns and their replacements, such as `matmul_fuse_pattern` and `matmul_replacement`, which fuse multiple matrix multiplications into a single operation and then split the result. These patterns are registered using `register_replacement`, which allows the system to recognize and replace specific operation sequences during execution. This optimization is crucial for improving computational efficiency.

```python
def addmm_patterns_init():
    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    def check_concat_weights(match):
        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]

        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")

            equal_shape_inputs.append(bias_inputs)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/freezing_patterns.py" line="158">

---

### Registering Replacement Patterns

The `register_replacement` function is called to register the `matmul_fuse_pattern` and `matmul_replacement`. This registration allows the system to replace sequences of matrix multiplications with a more efficient fused operation.

```python
    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/freezing_patterns.py" line="124">

---

### Checking Concatenated Weights

The `check_concat_weights` function checks if the weights to be concatenated have the same shape. This ensures that the concatenation and subsequent operations are valid and can be performed without errors.

```python
    def check_concat_weights(match):
        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]

        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")

            equal_shape_inputs.append(bias_inputs)

        for equal_shape_group in equal_shape_inputs:
            inps = [match.kwargs[name] for name in equal_shape_group]

            if not all(
                inp.op == "get_attr"
                and inp.meta["val"].shape == inps[0].meta["val"].shape
                for inp in inps
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/fx_passes/post_grad.py" line="974">

---

## addmm Function

The `addmm` function replaces a pattern of matrix addition and multiplication with a more optimized version. It uses the `replace_by_example` method to perform this replacement, ensuring that the optimized operation is used during execution.

```python
def addmm(match, mat1, mat2, *, inp):
    def repl(inp, mat1, mat2):
        return aten.addmm(inp, mat1, mat2)

    match.replace_by_example(repl, [inp, mat1, mat2])
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/pattern_matcher.py" line="224">

---

## Replacing by Example

The `replace_by_example` function replaces a given pattern with a specified replacement function. It traces the replacement function to create a new computational graph, which is then used to replace the original pattern. This method is essential for optimizing specific sequences of operations.

```python
    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: Optional[TraceFn] = None,
        run_dce: bool = True,
    ) -> None:
        from torch._inductor.virtualized import V

        context = V.fake_mode if V.fake_mode is not None else contextlib.nullcontext

        with context:
            if trace_fn is None:
                trace_fn = functools.partial(fwd_only, run_dce=run_dce)
            replacement = trace_fn(
                replacement_fn, torch.fx.map_arg(args, lambda arg: arg.meta["val"])
            )
            ReplacementPatternEntry.replace_with_graph(
                self,
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      5dd00346c8f29cefda91fc3dbe3db612537d60682f6450f5df325f8b0cbede3b(register_replacement):::mainFlowStyle --> fb2a894ced1d21f5513b07111a77e9a3b1457976bbb1246c3f1e68dbba73009f(gen_pattern):::mainFlowStyle

fb2a894ced1d21f5513b07111a77e9a3b1457976bbb1246c3f1e68dbba73009f(gen_pattern):::mainFlowStyle --> 3192db0a19b37cdee5ce1d03ca49edd9406e9385950b7b77164328a7e0b5f559(trace_fn):::mainFlowStyle

3192db0a19b37cdee5ce1d03ca49edd9406e9385950b7b77164328a7e0b5f559(trace_fn):::mainFlowStyle --> 46d0c3ff106c08d7937702fd73ddbb320c4c519be87b38dcedc625835f91b8a8(make_dual):::mainFlowStyle

46d0c3ff106c08d7937702fd73ddbb320c4c519be87b38dcedc625835f91b8a8(make_dual):::mainFlowStyle --> de7c515ae4b8c45003b6984425b592ce24c09e8a3d9d2022e0747b452ffd3b39(_make_dual):::mainFlowStyle


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_inductor/pattern_matcher.py" line="1220">

---

## register_replacement

The function `register_replacement` is responsible for registering a replacement pattern in the pattern matcher. It ensures that shapes are correctly matched by initially ignoring certain types like integers.

```python
    def check_fn(match: Match) -> bool:
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/pattern_matcher.py" line="1274">

---

The `search_fn_new` function within `register_replacement` adjusts the arguments for the search function, ensuring that the correct subset of arguments is passed during the pattern matching process.

```python
                    def search_fn_new(*args_new: Any) -> Any:
                        return search_fn(*args_new[len(args_new) - len(args) :])
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_inductor/pattern_matcher.py" line="1326">

---

The `normalize_args` function within `register_replacement` normalizes the arguments by extracting them from the keyword arguments, ensuring that the pattern matcher has a consistent view of the arguments.

```python
    def normalize_args(**kwargs: Any) -> List[Any]:
        args = []
        for name in argnames_static:
            args.append(kwargs.pop(name))
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
