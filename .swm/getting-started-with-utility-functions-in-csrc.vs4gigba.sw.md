---
title: Getting started with Utility Functions in Csrc
---
# Getting Started with Utility Functions in Csrc

Utils in Csrc refers to a collection of utility functions and classes that provide various helper functionalities. These utilities include functions for handling Python bindings, managing CUDA availability, and working with structured sequences. They also provide support for benchmarking, variadic templates, and nested tensor construction. The utils are essential for extending PyTorch's capabilities and ensuring smooth integration with Python and C++ code.

## Python Tuple Handling

This file contains utility functions for handling Python tuples, such as packing and unpacking <SwmToken path="torch/_C/_onnx.pyi" pos="15:1:1" line-data="    INT64 = ...">`INT64`</SwmToken> arrays into Python tuples.

<SwmSnippet path="/torch/csrc/utils/python_tuples.h" line="1">

---

The function <SwmToken path="torch/csrc/utils/python_tuples.h" pos="8:4:4" line-data="inline void THPUtils_packInt64Array(">`THPUtils_packInt64Array`</SwmToken> packs an array of <SwmToken path="torch/_C/_onnx.pyi" pos="15:1:1" line-data="    INT64 = ...">`INT64`</SwmToken> values into a Python tuple. This is useful for converting C++ data structures into Python-compatible formats.

```c
#pragma once

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

inline void THPUtils_packInt64Array(
    PyObject* tuple,
    size_t size,
    const int64_t* sizes) {
  for (size_t i = 0; i != size; ++i) {
    PyObject* i64 = THPUtils_packInt64(sizes[i]);
    if (!i64) {
      throw python_error();
    }
    PyTuple_SET_ITEM(tuple, i, i64);
  }
}

inline PyObject* THPUtils_packInt64Array(size_t size, const int64_t* sizes) {
```

---

</SwmSnippet>

## Initialization Functions

This file includes functions for initializing various components, such as throughput benchmark bindings.

<SwmSnippet path="/torch/csrc/utils/init.h" line="1">

---

The function <SwmToken path="torch/csrc/utils/init.h" pos="7:2:2" line-data="void initThroughputBenchmarkBindings(PyObject* module);">`initThroughputBenchmarkBindings`</SwmToken> initializes the throughput benchmark bindings in the provided Python module.

```c
#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module);

} // namespace torch::throughput_benchmark
```

---

</SwmSnippet>

## Type Checking Functions

This file provides functions for checking if the output types of tensors match the expected types.

<SwmSnippet path="/torch/csrc/utils/out_types.h" line="1">

---

The function <SwmToken path="torch/csrc/utils/out_types.h" pos="7:4:4" line-data="TORCH_API void check_out_type_matches(">`check_out_type_matches`</SwmToken> ensures that the output tensor's type matches the expected scalar type, layout, and device.

```c
#pragma once

#include <ATen/core/Tensor.h>

namespace torch::utils {

TORCH_API void check_out_type_matches(
    const at::Tensor& result,
    std::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool device_is_none);

}
```

---

</SwmSnippet>

## Main Functions

There are several main functions in this folder. Some of them are <SwmToken path="torch/csrc/utils/python_numbers.h" pos="28:5:5" line-data="inline PyObject* THPUtils_packUInt32(uint32_t value) {">`THPUtils_packUInt32`</SwmToken>, <SwmToken path="torch/csrc/utils/python_numbers.h" pos="32:5:5" line-data="inline PyObject* THPUtils_packUInt64(uint64_t value) {">`THPUtils_packUInt64`</SwmToken>, <SwmToken path="torch/csrc/utils/python_numbers.h" pos="87:4:4" line-data="inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {">`THPUtils_unpackUInt32`</SwmToken>, and <SwmToken path="torch/csrc/utils/python_numbers.h" pos="98:4:4" line-data="inline uint64_t THPUtils_unpackUInt64(PyObject* obj) {">`THPUtils_unpackUInt64`</SwmToken>. We will dive a little into <SwmToken path="torch/csrc/utils/python_numbers.h" pos="28:5:5" line-data="inline PyObject* THPUtils_packUInt32(uint32_t value) {">`THPUtils_packUInt32`</SwmToken> and <SwmToken path="torch/csrc/utils/python_numbers.h" pos="87:4:4" line-data="inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {">`THPUtils_unpackUInt32`</SwmToken>.

### <SwmToken path="torch/csrc/utils/python_numbers.h" pos="28:5:5" line-data="inline PyObject* THPUtils_packUInt32(uint32_t value) {">`THPUtils_packUInt32`</SwmToken>

The <SwmToken path="torch/csrc/utils/python_numbers.h" pos="28:5:5" line-data="inline PyObject* THPUtils_packUInt32(uint32_t value) {">`THPUtils_packUInt32`</SwmToken> function converts a 32-bit unsigned integer into a Python object using <SwmToken path="torch/csrc/utils/python_numbers.h" pos="29:3:3" line-data="  return PyLong_FromUnsignedLong(value);">`PyLong_FromUnsignedLong`</SwmToken>. This function is used in various parts of the codebase to handle 32-bit unsigned integers in Python.

<SwmSnippet path="/torch/csrc/utils/python_numbers.h" line="28">

---

The function <SwmToken path="torch/csrc/utils/python_numbers.h" pos="28:5:5" line-data="inline PyObject* THPUtils_packUInt32(uint32_t value) {">`THPUtils_packUInt32`</SwmToken> converts a 32-bit unsigned integer into a Python object.

```c
inline PyObject* THPUtils_packUInt32(uint32_t value) {
  return PyLong_FromUnsignedLong(value);
}
```

---

</SwmSnippet>

### <SwmToken path="torch/csrc/utils/python_numbers.h" pos="87:4:4" line-data="inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {">`THPUtils_unpackUInt32`</SwmToken>

The <SwmToken path="torch/csrc/utils/python_numbers.h" pos="87:4:4" line-data="inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {">`THPUtils_unpackUInt32`</SwmToken> function converts a Python object back into a 32-bit unsigned integer. It uses <SwmToken path="torch/csrc/utils/python_numbers.h" pos="88:9:9" line-data="  unsigned long value = PyLong_AsUnsignedLong(obj);">`PyLong_AsUnsignedLong`</SwmToken> and includes error handling to ensure the value fits within the range of a 32-bit unsigned integer.

<SwmSnippet path="/torch/csrc/utils/python_numbers.h" line="87">

---

The function <SwmToken path="torch/csrc/utils/python_numbers.h" pos="87:4:4" line-data="inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {">`THPUtils_unpackUInt32`</SwmToken> converts a Python object back into a 32-bit unsigned integer and includes error handling.

```c
inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {
  unsigned long value = PyLong_AsUnsignedLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Overflow when unpacking unsigned long");
  }
  return (uint32_t)value;
}
```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
