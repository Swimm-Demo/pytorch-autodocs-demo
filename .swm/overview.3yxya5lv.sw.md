---
title: Overview
---
## Main Components

### Torch

Torch is a core library in PyTorch that provides a wide range of functionalities for tensor computation, neural network building, and various utilities essential for machine learning and deep learning tasks.

&nbsp;

- **Misc**
  - <SwmLink doc-title="Overview of Passes in FX">[Overview of Passes in FX](/.swm/overview-of-passes-in-fx.qqd71w1j.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of Neural Network Utilities">[Overview of Neural Network Utilities](/.swm/overview-of-neural-network-utilities.faoj4oy2.sw.md)</SwmLink>
  - **Flows**
    - <SwmLink doc-title="Overview of the convert_pt2e Process">[Overview of the convert_pt2e Process](/.swm/overview-of-the-convert_pt2e-process.h4n1kro7.sw.md)</SwmLink>
    - <SwmLink doc-title="Switching Exported Model Modes">[Switching Exported Model Modes](/.swm/switching-exported-model-modes.0kq43tar.sw.md)</SwmLink>
    - <SwmLink doc-title="Training Mode Toggle">[Training Mode Toggle](/.swm/training-mode-toggle.vvaabh56.sw.md)</SwmLink>
- **NN**
  - **Modules**
    - <SwmLink doc-title="Getting Started with PyTorch Modules">[Getting Started with PyTorch Modules](/.swm/getting-started-with-pytorch-modules.5l1pi9l4.sw.md)</SwmLink>
    - <SwmLink doc-title="The Module class in Detail">[The Module class in Detail](/.swm/the-module-class-in-detail.jmpbc.sw.md)</SwmLink>
- **Csrc**
  - <SwmLink doc-title="Getting started with Utility Functions in Csrc">[Getting started with Utility Functions in Csrc](/.swm/getting-started-with-utility-functions-in-csrc.vs4gigba.sw.md)</SwmLink>
  - <SwmLink doc-title="Basic Concepts of Dynamo in Csrc">[Basic Concepts of Dynamo in Csrc](/.swm/basic-concepts-of-dynamo-in-csrc.plq966vw.sw.md)</SwmLink>
  - **Profiler**
    - <SwmLink doc-title="Exploring the Unwind Process">[Exploring the Unwind Process](/.swm/exploring-the-unwind-process.nswaqbub.sw.md)</SwmLink>
  - **Api**
    - <SwmLink doc-title="Exploring Neural Network Module">[Exploring Neural Network Module](/.swm/exploring-neural-network-module.cld6qxtf.sw.md)</SwmLink>
  - **Nn**
    - <SwmLink doc-title="Overview of Functional Operators">[Overview of Functional Operators](/.swm/overview-of-functional-operators.g9o3de2j.sw.md)</SwmLink>
  - **Lazy**
    - <SwmLink doc-title="Exploring Lazy Execution">[Exploring Lazy Execution](/.swm/exploring-lazy-execution.z6jzku6s.sw.md)</SwmLink>
    - <SwmLink doc-title="Introduction to TorchScript Backend">[Introduction to TorchScript Backend](/.swm/introduction-to-torchscript-backend.twioe0xc.sw.md)</SwmLink>
    - <SwmLink doc-title="Lazy Core in PyTorch">[Lazy Core in PyTorch](/.swm/lazy-core-in-pytorch.fiuc9swg.sw.md)</SwmLink>
  - **Cuda**
    - <SwmLink doc-title="Basic Concepts of CUDA in CSRC">[Basic Concepts of CUDA in CSRC](/.swm/basic-concepts-of-cuda-in-csrc.oz9zhlgg.sw.md)</SwmLink>
    - <SwmLink doc-title="Overview of initCommMethods">[Overview of initCommMethods](/.swm/overview-of-initcommmethods.eesf23mt.sw.md)</SwmLink>
  - **Inductor**
    - <SwmLink doc-title="Basic Concepts of Inductor in Csrc">[Basic Concepts of Inductor in Csrc](/.swm/basic-concepts-of-inductor-in-csrc.zbwqms9o.sw.md)</SwmLink>
    - <SwmLink doc-title="Aoti Torch in Inductor">[Aoti Torch in Inductor](/.swm/aoti-torch-in-inductor.i3e1ks3j.sw.md)</SwmLink>
  - **Autograd**
    - <SwmLink doc-title="Autograd in PyTorch C++ Source Code">[Autograd in PyTorch C++ Source Code](/.swm/autograd-in-pytorch-c-source-code.gtoeehc1.sw.md)</SwmLink>
    - <SwmLink doc-title="Functions in Autograd">[Functions in Autograd](/.swm/functions-in-autograd.64c8ejok.sw.md)</SwmLink>
    - <SwmLink doc-title="Handling RPC Messages">[Handling RPC Messages](/.swm/handling-rpc-messages.zjlfl5h8.sw.md)</SwmLink>
  - **Distributed**
    - <SwmLink doc-title="Introduction to Distributed Autograd">[Introduction to Distributed Autograd](/.swm/introduction-to-distributed-autograd.17h104qk.sw.md)</SwmLink>
    - <SwmLink doc-title="Exploring Distributed RPC">[Exploring Distributed RPC](/.swm/exploring-distributed-rpc.qvay01mg.sw.md)</SwmLink>
    - **C10d**
      - <SwmLink doc-title="Introduction to C10d in Distributed">[Introduction to C10d in Distributed](/.swm/introduction-to-c10d-in-distributed.lio9tnjt.sw.md)</SwmLink>
      - <SwmLink doc-title="Getting started with MPI Process Group in c10d">[Getting started with MPI Process Group in c10d](/.swm/getting-started-with-mpi-process-group-in-c10d.fcq5nldh.sw.md)</SwmLink>
      - <SwmLink doc-title="Overview of Gloo Process Group in c10d">[Overview of Gloo Process Group in c10d](/.swm/overview-of-gloo-process-group-in-c10d.q92788ub.sw.md)</SwmLink>
      - <SwmLink doc-title="Getting started with Unified Communication X Process Group">[Getting started with Unified Communication X Process Group](/.swm/getting-started-with-unified-communication-x-process-group.o6brl4vy.sw.md)</SwmLink>
      - <SwmLink doc-title="Initialization in C10d">[Initialization in C10d](/.swm/initialization-in-c10d.88oi7dm1.sw.md)</SwmLink>
      - <SwmLink doc-title="Getting Started with Gradient Reducer">[Getting Started with Gradient Reducer](/.swm/getting-started-with-gradient-reducer.7xzu3ukp.sw.md)</SwmLink>
      - <SwmLink doc-title="Basic concepts of TCP Store Backend in C10d">[Basic concepts of TCP Store Backend in C10d](/.swm/basic-concepts-of-tcp-store-backend-in-c10d.8gjqrcy9.sw.md)</SwmLink>
      - **NCCL Process Group**
        - <SwmLink doc-title="NCCL Process Group in C10d">[NCCL Process Group in C10d](/.swm/nccl-process-group-in-c10d.hpr557qb.sw.md)</SwmLink>
        - <SwmLink doc-title="NCCL Error Checking Process">[NCCL Error Checking Process](/.swm/nccl-error-checking-process.e35x8gc1.sw.md)</SwmLink>
  - **Jit**
    - <SwmLink doc-title="Exploring Intermediate Representation in JIT">[Exploring Intermediate Representation in JIT](/.swm/exploring-intermediate-representation-in-jit.gkp8i08x.sw.md)</SwmLink>
    - <SwmLink doc-title="Overview of Frontend in Jit">[Overview of Frontend in Jit](/.swm/overview-of-frontend-in-jit.s8illvlu.sw.md)</SwmLink>
    - <SwmLink doc-title="Introduction to JIT Backends">[Introduction to JIT Backends](/.swm/introduction-to-jit-backends.tp4av7pm.sw.md)</SwmLink>
    - <SwmLink doc-title="Python Integration in JIT">[Python Integration in JIT](/.swm/python-integration-in-jit.qyhpgfxa.sw.md)</SwmLink>
    - **Runtime**
      - <SwmLink doc-title="Exploring Static Values in Runtime">[Exploring Static Values in Runtime](/.swm/exploring-static-values-in-runtime.6pl42yeg.sw.md)</SwmLink>
      - <SwmLink doc-title="Overview of FuseTensorExprs">[Overview of FuseTensorExprs](/.swm/overview-of-fusetensorexprs.5d1ol7sn.sw.md)</SwmLink>
    - **Flows**
      - <SwmLink doc-title="Inlining Process Overview">[Inlining Process Overview](/.swm/inlining-process-overview.7eoikrem.sw.md)</SwmLink>
      - <SwmLink doc-title="Overview of the parseIR Function">[Overview of the parseIR Function](/.swm/overview-of-the-parseir-function.zpsfy1q0.sw.md)</SwmLink>
    - **Tensorexpr**
      - <SwmLink doc-title="Introduction to Tensor Expressions in JIT">[Introduction to Tensor Expressions in JIT](/.swm/introduction-to-tensor-expressions-in-jit.b6v1mrk4.sw.md)</SwmLink>
      - <SwmLink doc-title="Operators in Tensorexpr">[Operators in Tensorexpr](/.swm/operators-in-tensorexpr.l6lw1x96.sw.md)</SwmLink>
    - **Serialization**
      - <SwmLink doc-title="Overview of Serialization in Jit">[Overview of Serialization in Jit](/.swm/overview-of-serialization-in-jit.mxws3yo6.sw.md)</SwmLink>
      - <SwmLink doc-title="Deserializing JIT Module from Flatbuffer">[Deserializing JIT Module from Flatbuffer](/.swm/deserializing-jit-module-from-flatbuffer.sh33m5yu.sw.md)</SwmLink>
    - **Mobile**
      - <SwmLink doc-title="Basic Concepts of Mobile in Jit">[Basic Concepts of Mobile in Jit](/.swm/basic-concepts-of-mobile-in-jit.axqp55i7.sw.md)</SwmLink>
      - <SwmLink doc-title="Overview of Model Tracer in Mobile">[Overview of Model Tracer in Mobile](/.swm/overview-of-model-tracer-in-mobile.8p64kqr8.sw.md)</SwmLink>
    - **Codegen**
      - <SwmLink doc-title="Exploring Fuser in Codegen">[Exploring Fuser in Codegen](/.swm/exploring-fuser-in-codegen.37srh8ah.sw.md)</SwmLink>
      - <SwmLink doc-title="Basic Concepts of Onednn Integration">[Basic Concepts of Onednn Integration](/.swm/basic-concepts-of-onednn-integration.r35agmuw.sw.md)</SwmLink>
    - **Passes**
      - <SwmLink doc-title="Getting Started with Quantization in Passes">[Getting Started with Quantization in Passes](/.swm/getting-started-with-quantization-in-passes.8v0lzrbe.sw.md)</SwmLink>
      - <SwmLink doc-title="Basic Concepts of ONNX in Passes">[Basic Concepts of ONNX in Passes](/.swm/basic-concepts-of-onnx-in-passes.gsgc2w1l.sw.md)</SwmLink>
      - <SwmLink doc-title="Creating a Tensor Expression Operation">[Creating a Tensor Expression Operation](/.swm/creating-a-tensor-expression-operation.aaadyqwv.sw.md)</SwmLink>
    - **Flows**
      - <SwmLink doc-title="Initialization of JIT Bindings">[Initialization of JIT Bindings](/.swm/initialization-of-jit-bindings.cr8su2og.sw.md)</SwmLink>
      - <SwmLink doc-title="Optimizing a Computational Graph">[Optimizing a Computational Graph](/.swm/optimizing-a-computational-graph.83c2d2ab.sw.md)</SwmLink>
      - <SwmLink doc-title="Initializing Embedding Layer with Pre-trained Tensor">[Initializing Embedding Layer with Pre-trained Tensor](/.swm/initializing-embedding-layer-with-pre-trained-tensor.oa4oc11p.sw.md)</SwmLink>
      - <SwmLink doc-title="Triton Compilation Process">[Triton Compilation Process](/.swm/triton-compilation-process.b7p9mlce.sw.md)</SwmLink>
      - <SwmLink doc-title="Purpose of the warm_pool Function">[Purpose of the warm_pool Function](/.swm/purpose-of-the-warm_pool-function.16lylpat.sw.md)</SwmLink>
      - <SwmLink doc-title="Overview of _new_pool Function">[Overview of \_new_pool Function](/.swm/overview-of-_new_pool-function.5z11hdgy.sw.md)</SwmLink>
  - **Flows**
    - <SwmLink doc-title="Setting Up ONNX Bindings">[Setting Up ONNX Bindings](/.swm/setting-up-onnx-bindings.yrgaxq8i.sw.md)</SwmLink>
    - <SwmLink doc-title="Handling RPC Commands and Errors">[Handling RPC Commands and Errors](/.swm/handling-rpc-commands-and-errors.impwvw9l.sw.md)</SwmLink>
- **Flows**
  - <SwmLink doc-title="Converting Patterns to FX Graphs">[Converting Patterns to FX Graphs](/.swm/converting-patterns-to-fx-graphs.x51v18q1.sw.md)</SwmLink>
  - <SwmLink doc-title="Determining Tensor Size">[Determining Tensor Size](/.swm/determining-tensor-size.wua16vmt.sw.md)</SwmLink>
  - <SwmLink doc-title="Converting and Loading Optimizer State Dictionary">[Converting and Loading Optimizer State Dictionary](/.swm/converting-and-loading-optimizer-state-dictionary.bunt32s5.sw.md)</SwmLink>
  - <SwmLink doc-title="Profiling Process Overview">[Profiling Process Overview](/.swm/profiling-process-overview.j6jl4pye.sw.md)</SwmLink>
  - <SwmLink doc-title="Standard Deviation Calculation Flow">[Standard Deviation Calculation Flow](/.swm/standard-deviation-calculation-flow.lnzlzd3b.sw.md)</SwmLink>
  - <SwmLink doc-title="Compilation Process Overview">[Compilation Process Overview](/.swm/compilation-process-overview.xneyioh4.sw.md)</SwmLink>
  - <SwmLink doc-title="Tracing Supported Methods Flow">[Tracing Supported Methods Flow](/.swm/tracing-supported-methods-flow.tc7nm5xy.sw.md)</SwmLink>
  - <SwmLink doc-title="Compiling with Persistent Cache">[Compiling with Persistent Cache](/.swm/compiling-with-persistent-cache.nsnu9ga8.sw.md)</SwmLink>
  - <SwmLink doc-title="Compilation Process Overview">[Compilation Process Overview](/.swm/compilation-process-overview.k75dkutg.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of the __call__ Flow">[Overview of the \__call_\_ Flow](/.swm/overview-of-the-__call__-flow.ynkr3or1.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of the Inductor Function">[Overview of the Inductor Function](/.swm/overview-of-the-inductor-function.8mhnp3qq.sw.md)</SwmLink>
  - <SwmLink doc-title="Repro Run Flow">[Repro Run Flow](/.swm/repro-run-flow.cn3as6gr.sw.md)</SwmLink>
  - <SwmLink doc-title="Analyzing and Comparing Model Outputs">[Analyzing and Comparing Model Outputs](/.swm/analyzing-and-comparing-model-outputs.3kw9kzs8.sw.md)</SwmLink>
  - <SwmLink doc-title="Loading Arguments and Running Compilation">[Loading Arguments and Running Compilation](/.swm/loading-arguments-and-running-compilation.25n9b32h.sw.md)</SwmLink>
  - <SwmLink doc-title="Handling Compilation Failures">[Handling Compilation Failures](/.swm/handling-compilation-failures.dd4vctzm.sw.md)</SwmLink>
  - <SwmLink doc-title="Debugging Graph Compilation">[Debugging Graph Compilation](/.swm/debugging-graph-compilation.l3v9qua0.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of checkGraphModeOp">[Overview of checkGraphModeOp](/.swm/overview-of-checkgraphmodeop.osk7zlac.sw.md)</SwmLink>
  - <SwmLink doc-title="Model Freezing Process">[Model Freezing Process](/.swm/model-freezing-process.o7k3h4ay.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of train_convnext_example">[Overview of train_convnext_example](.swm/overview-of-train_convnext_example.7egh2xb7.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of the _apply Function">[Overview of the \_apply Function](/.swm/overview-of-the-_apply-function.cz2dkcnl.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of aot_dispatch_autograd">[Overview of aot_dispatch_autograd](.swm/overview-of-aot_dispatch_autograd.tfnc9hc2.sw.md)</SwmLink>
  - <SwmLink doc-title="Converting a Model to NNAPI">[Converting a Model to NNAPI](/.swm/converting-a-model-to-nnapi.0zt4zne6.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of the Hessian Function">[Overview of the Hessian Function](/.swm/overview-of-the-hessian-function.ahcy61q7.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of fw_compiler_base">[Overview of fw_compiler_base](.swm/overview-of-fw_compiler_base.zy5uyk0e.sw.md)</SwmLink>
  - <SwmLink doc-title="Handling Real Inputs with Deferred Execution">[Handling Real Inputs with Deferred Execution](/.swm/handling-real-inputs-with-deferred-execution.qjkc5g1d.sw.md)</SwmLink>
  - <SwmLink doc-title="Gradient Checking for Sparse Tensors">[Gradient Checking for Sparse Tensors](/.swm/gradient-checking-for-sparse-tensors.7iqome21.sw.md)</SwmLink>
  - <SwmLink doc-title="Gradient Checking with Sparse Support">[Gradient Checking with Sparse Support](/.swm/gradient-checking-with-sparse-support.k87x4ufq.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of the _test_quantizer Function">[Overview of the \_test_quantizer Function](/.swm/overview-of-the-_test_quantizer-function.hasdbpyk.sw.md)</SwmLink>
  - <SwmLink doc-title="Debugging the Compilation Process">[Debugging the Compilation Process](/.swm/debugging-the-compilation-process.9uqnk6a0.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of simple_ts_compile">[Overview of simple_ts_compile](.swm/overview-of-simple_ts_compile.a0pi84fx.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of _fast_gradcheck Function">[Overview of \_fast_gradcheck Function](/.swm/overview-of-_fast_gradcheck-function.wibv8mib.sw.md)</SwmLink>
  - <SwmLink doc-title="Handling Tensor Mutations in Triton Kernel">[Handling Tensor Mutations in Triton Kernel](/.swm/handling-tensor-mutations-in-triton-kernel.795rvuez.sw.md)</SwmLink>
  - <SwmLink doc-title="Verifying Optimizer State Dictionary">[Verifying Optimizer State Dictionary](/.swm/verifying-optimizer-state-dictionary.wmq2d5zv.sw.md)</SwmLink>
  - <SwmLink doc-title="Optimizing Attribute Tracing in nn.Module">[Optimizing Attribute Tracing in nn.Module](/.swm/optimizing-attribute-tracing-in-nnmodule.3ujuez3z.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of Codegen Node">[Overview of Codegen Node](/.swm/overview-of-codegen-node.q90e4yjb.sw.md)</SwmLink>
  - <SwmLink doc-title="Prediction Process Overview">[Prediction Process Overview](/.swm/prediction-process-overview.q69j4l8h.sw.md)</SwmLink>
  - <SwmLink doc-title="Quantization Process Overview">[Quantization Process Overview](/.swm/quantization-process-overview.10rutu09.sw.md)</SwmLink>
  - <SwmLink doc-title="Generating a Quantized Model">[Generating a Quantized Model](/.swm/generating-a-quantized-model.ami9p9e2.sw.md)</SwmLink>
  - <SwmLink doc-title="Reporting Model Exportability">[Reporting Model Exportability](/.swm/reporting-model-exportability.utn1smep.sw.md)</SwmLink>
  - <SwmLink doc-title="Running a Reproducibility Test">[Running a Reproducibility Test](/.swm/running-a-reproducibility-test.2puzxj0x.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of wrapper_fn">[Overview of wrapper_fn](.swm/overview-of-wrapper_fn.4i5qtom2.sw.md)</SwmLink>
  - <SwmLink doc-title="Saving and Loading Model States with FSDP">[Saving and Loading Model States with FSDP](/.swm/saving-and-loading-model-states-with-fsdp.eo5dypeg.sw.md)</SwmLink>
  - <SwmLink doc-title="Overview of _slow_gradcheck">[Overview of \_slow_gradcheck](.swm/overview-of-_slow_gradcheck.sfavb38d.sw.md)</SwmLink>
- **Classes**
  - <SwmLink doc-title="The VariableTracker class">[The VariableTracker class](/.swm/the-variabletracker-class.7trh1.sw.md)</SwmLink>
  - <SwmLink doc-title="The IterDataPipe class">[The IterDataPipe class](/.swm/the-iterdatapipe-class.6wjfg.sw.md)</SwmLink>

### Flows

- <SwmLink doc-title="Combining Input Tensor with Mask">[Combining Input Tensor with Mask](/.swm/combining-input-tensor-with-mask.680v0enj.sw.md)</SwmLink>
- <SwmLink doc-title="Main Function Flow">[Main Function Flow](/.swm/main-function-flow.grv4yeag.sw.md)</SwmLink>
- <SwmLink doc-title="Data Copying Process">[Data Copying Process](/.swm/data-copying-process.h9a3mst3.sw.md)</SwmLink>
- <SwmLink doc-title="Convolution Flow Overview">[Convolution Flow Overview](/.swm/convolution-flow-overview.z1138xqk.sw.md)</SwmLink>
- <SwmLink doc-title="Convolution Backward Pass Overview">[Convolution Backward Pass Overview](/.swm/convolution-backward-pass-overview.nxvjx107.sw.md)</SwmLink>

## Classes

- <SwmLink doc-title="The object class">[The object class](.swm/the-object-class.g40w6.sw.md)</SwmLink>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
