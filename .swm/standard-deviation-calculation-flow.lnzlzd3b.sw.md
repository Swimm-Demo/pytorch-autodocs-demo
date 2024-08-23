---
title: Standard Deviation Calculation Flow
---
This document provides an overview of the flow for calculating the standard deviation of a masked tensor. It includes a high-level diagram and a detailed breakdown of the key functions involved in the process.

The flow for calculating the standard deviation starts with the `std` function, which calls `_std_var` to compute the variance. The `_std_var` function handles different data types, tensor layouts, and masks. It calculates the mean, subtracts it from the input, squares the result, and sums it up to get the variance. If `take_sqrt` is `True`, it returns the square root of the variance, which is the standard deviation. The `_input_mask` function ensures the mask is compatible with the input tensor, and `_combine_input_and_mask` applies the mask to the input. The `_sparse_coo_scatter_reduction_helper` handles reduction operations for sparse COO tensors, and `_where` ensures the correct values for masked-in elements.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      subgraph torchmaskedopspy["torch/masked/_ops.py"]
b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle --> 856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle --> 559cff12557e16ce617c0a404e722ef48a6cf117a300face99bcb4c9ccf85679(_input_mask)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle --> e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle --> 8cc9950ef5cc1fbb02c45ac968135ec5e500fff3d9125eb98c3300fdfa2d8d8b(_combine_input_and_mask)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle --> a85658eb0aa0bb4a35fc620d7547e097d76a4dad875841a2a51fe231d34fc287(_sparse_coo_scatter_reduction_helper):::mainFlowStyle
end


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
      subgraph torchmaskedopspy["torch/masked/_ops.py"]
b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle --> 856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle --> 559cff12557e16ce617c0a404e722ef48a6cf117a300face99bcb4c9ccf85679(_input_mask)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
856baa4ee56ae46089ddacdba603e43db2b27f1204a33d005282a847d5d4d03b(_std_var):::mainFlowStyle --> e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle --> juv8k(...)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/masked/_ops.py" line="1675">

---

## Calculating the standard deviation

The `std` function calculates the standard deviation of the input tensor. It calls the `_std_var` function with the `take_sqrt` parameter set to `True`, indicating that the square root of the variance should be taken to get the standard deviation.

```python
def std(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    unbiased: Optional[bool] = None,
    *,
    correction: Optional[int] = None,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
The identity value of sample standard deviation operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    return _std_var(
```

---

</SwmSnippet>

<SwmSnippet path="/torch/masked/_ops.py" line="1571">

---

## Computing variance

The `_std_var` function computes the variance of the input tensor. It handles various cases such as different data types, tensor layouts, and the presence of a mask. The function calculates the mean, subtracts it from the input, squares the result, and then sums it up to get the total variance. If `take_sqrt` is `True`, it returns the square root of the variance, which is the standard deviation.

```python
def _std_var(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims,
    unbiased: Optional[bool],
    *,
    correction_opt: Optional[Union[int, float]],
    keepdim: Optional[bool],
    dtype: Optional[DType],
    mask: Optional[Tensor],
    take_sqrt: Optional[bool],
) -> Tensor:
    assert (
        unbiased is None or correction_opt is None
    ), "Only one of unbiased and correction may be given"
    correction = 1.0
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    if correction_opt is not None:
        correction = sym_float(correction_opt)

    if dtype is None:
```

---

</SwmSnippet>

<SwmSnippet path="/torch/masked/_ops.py" line="858">

---

### Handling input masks

The `_input_mask` function ensures that the mask tensor is compatible with the input tensor. It adjusts the shape, layout, and data type of the mask to match the input tensor, ensuring that the mask can be correctly applied during the variance computation.

```python
def _input_mask(input: Union[Tensor, MaskedTensor], *args, **kwargs) -> Tensor:
    """Return canonical input mask.

    A canonical input mask is defined as a boolean mask tensor that
    shape and layout matches with the shape and the layout of the
    input.

    The canonical input mask is computed from the :attr:`mask` tensor
    content to meet the following criteria:

    1. The shape of the canonical input mask is the same as the shape
       of :attr:`input` tensor. If the mask tensor has a smaller shape
       than the shape of the :attr:`input`, broadcasting rules will be
       applied. Downcasting of mask is not supported.

    2. The layout of the canonical input mask is the same as the
       layout of the :attr:`input` tensor. If the mask has different
       layout, it will be converted to the expected layout.  In the
       case of sparse COO layout, the canonical input mask will be
       coalesced.

```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchmaskedopspy["torch/masked/_ops.py"]
e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle --> 8cc9950ef5cc1fbb02c45ac968135ec5e500fff3d9125eb98c3300fdfa2d8d8b(_combine_input_and_mask)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
e35ed3b3b7e0a55b73c98094ebc01a8923e92e83f9e2f13f20f07591d27895a5(sum):::mainFlowStyle --> a85658eb0aa0bb4a35fc620d7547e097d76a4dad875841a2a51fe231d34fc287(_sparse_coo_scatter_reduction_helper):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
8cc9950ef5cc1fbb02c45ac968135ec5e500fff3d9125eb98c3300fdfa2d8d8b(_combine_input_and_mask) --> 32574252951c659c70f20b28176f8122bbbee64895117dfe00746c2995cdb823(helper)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
8cc9950ef5cc1fbb02c45ac968135ec5e500fff3d9125eb98c3300fdfa2d8d8b(_combine_input_and_mask) --> 09104fe63730ec071e41db6e90a8709be9e49da3a3ba63b3acad0499e2bd6e8f(_where)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
8cc9950ef5cc1fbb02c45ac968135ec5e500fff3d9125eb98c3300fdfa2d8d8b(_combine_input_and_mask) --> 559cff12557e16ce617c0a404e722ef48a6cf117a300face99bcb4c9ccf85679(_input_mask)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/masked/_ops.py" line="989">

---

## Combining Input and Mask

The function `_combine_input_and_mask` combines the input tensor with a mask. It ensures that masked-out elements are handled correctly by either returning the input directly if no mask is provided or applying the mask to the input. This is crucial for operations that need to consider masked elements, such as the `sum` function.

```python
def _combine_input_and_mask(
    op, input: Union[MaskedTensor, Tensor], mask, *args
) -> Tensor:
    def helper(input, mask):
        if mask is None:
            return input
        canonical_mask = _input_mask(input, mask=mask)
        if callable(op):
            fill_value = _reduction_identity(op.__name__, input, *args)
            return _where(canonical_mask, input, fill_value)
        else:
            raise ValueError(
                f"_combine_input_and_mask expected masked operation (got {type(op).__name__} object)"
            )

    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, mask):
            """Return input with masked-out elements eliminated for the given operations."""
            ctx.save_for_backward(mask)

```

---

</SwmSnippet>

<SwmSnippet path="/torch/masked/_ops.py" line="992">

---

### Helper Function

The `helper` function within `_combine_input_and_mask` is responsible for applying the mask to the input tensor. It uses `_input_mask` to get a canonical mask and `_where` to combine the input and mask, filling masked-out elements with a reduction identity value.

```python
    def helper(input, mask):
        if mask is None:
            return input
        canonical_mask = _input_mask(input, mask=mask)
        if callable(op):
            fill_value = _reduction_identity(op.__name__, input, *args)
            return _where(canonical_mask, input, fill_value)
        else:
            raise ValueError(
                f"_combine_input_and_mask expected masked operation (got {type(op).__name__} object)"
            )
```

---

</SwmSnippet>

<SwmSnippet path="/torch/masked/_ops.py" line="608">

---

## Sparse COO Scatter Reduction Helper

The function `_sparse_coo_scatter_reduction_helper` handles reduction operations for sparse COO tensors. It supports various reduction operations like `sum`, `prod`, `amax`, and `amin`. The function ensures that the reduction is applied correctly across both sparse and dense dimensions, handling edge cases like empty dimensions and dtype promotion.

```python
def _sparse_coo_scatter_reduction_helper(
    op,
    mask_input: Tensor,
    dims: Tuple[int, ...],
    keepdim: bool,
    dtype: Optional[DType] = None,
) -> Tensor:
    reduce = op.__name__
    valid_reductions = ["sum", "prod", "amax", "amin"]
    if reduce not in valid_reductions:
        raise ValueError(
            f"op must be one of {' '.join(valid_reductions)}, but got {reduce} instead"
        )

    output_dtype = dtype
    values, indices = mask_input._values(), mask_input._indices()
    input_dims = mask_input.dim()
    num_sparse_dims = mask_input.sparse_dim()
    reduced_sparse_dims = []
    retained_sparse_dims = []
    reduced_dense_dims = []
```

---

</SwmSnippet>

<SwmSnippet path="/torch/masked/_ops.py" line="822">

---

## Where Function

The `_where` function is a specialized version of `torch.where` that supports sparse inputs. It ensures that the resulting tensor maintains the correct values for masked-in elements and replaces masked-out elements with a specified fill value. This function is essential for operations that need to handle sparse tensors correctly.

```python
def _where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """torch.where with sparse inputs support.

    _where implements the following invariant:

      _where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.
```

---

</SwmSnippet>

# Where is this flow used?

This flow is used multiple times in the codebase as represented in the following diagram:

(Note - these are only some of the entry points of this flow)

```mermaid
graph TD;
      subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
227eaee48f790cb276fab7c3bb5c4aed07dc34a153c81931559b3ec413b1cb32(main):::rootsStyle --> b7fe36bb9fb1f39a71a0ecf73fb6ce9fa2acd7118f811f3b54df5f4569efd3cd(get_df)
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
b7fe36bb9fb1f39a71a0ecf73fb6ce9fa2acd7118f811f3b54df5f4569efd3cd(get_df) --> c732fbca97c05ba57bc38733aac53c3393c8629daf0aa5a802fce5d50552fe67(process_data)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
c732fbca97c05ba57bc38733aac53c3393c8629daf0aa5a802fce5d50552fe67(process_data) --> b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
e00a13529140069ac3788eaecba3cbdc757dfbcd58a4cf971dbb26e9afb2ec6e(generate_heuristic):::rootsStyle --> 742d455c98a960d95c66a4ffc07ef7dbe1921a7867710a9f016ae377c8b4fa2c(main)
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
742d455c98a960d95c66a4ffc07ef7dbe1921a7867710a9f016ae377c8b4fa2c(main) --> 066b1aa906a0f78b15fcf417e78ab6166720732bd0959ff949b8391160856d04(prepare_datasets)
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
066b1aa906a0f78b15fcf417e78ab6166720732bd0959ff949b8391160856d04(prepare_datasets) --> db1ca04f1dfa21917ad04700e361b18e34de36cf4dbd248eee9473b5e9be1f5e(add_real_datasets)
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
db1ca04f1dfa21917ad04700e361b18e34de36cf4dbd248eee9473b5e9be1f5e(add_real_datasets) --> 4e5e6e9ce572e334251bcfe093807b802063da8b7c3c62d4dcb00713a14778f1(get_df)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
4e5e6e9ce572e334251bcfe093807b802063da8b7c3c62d4dcb00713a14778f1(get_df) --> b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle
end

subgraph torchgenautoheuristic["torchgen/_autoheuristic"]
33e6b8040800789a08f0b11e0f33712bbec01d99a1a3e8a258c7a18af7952fb1(add_real_datasets):::rootsStyle --> 4e5e6e9ce572e334251bcfe093807b802063da8b7c3c62d4dcb00713a14778f1(get_df)
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
662bbe35b3b939182994444a101965403c8fc70b543582e385a651865779b2f3(calculate_stats):::rootsStyle --> b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle
end

subgraph torchmaskedopspy["torch/masked/_ops.py"]
662bbe35b3b939182994444a101965403c8fc70b543582e385a651865779b2f3(calculate_stats):::rootsStyle --> b7ebf82481223584c9fcec38c07f91af240a48ee7d0b0f69e4c9b6d4dd2c9bef(std):::mainFlowStyle
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
