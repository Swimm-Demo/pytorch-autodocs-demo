---
title: Handling RPC Commands and Errors
---
This document provides an overview of how RPC commands are processed and managed, especially focusing on error handling. It outlines the general flow of processing RPC commands, capturing and handling errors, and ensuring the integrity of distributed operations.

The process starts with attempting to handle an RPC command. If an error occurs, it is captured and managed appropriately. The system ensures that any Python-related errors are handled with the necessary interpreter locks. For other types of errors, it captures the error details and returns them. This ensures that the system can continue operating smoothly even when errors occur.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
310082d993a7c55b78c5e17623d267c8b5195c68c44ced95532408a33acc6486(processRpcWithErrors):::mainFlowStyle --> 4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> fe13cd6f8c31c181e7e9e2d4abc935cde862bfa7910bb9f725510e051c46bdfe(processCleanupAutogradContextReq)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> accb07d8d1032d75dcbc1a0844da633bd21c1c06196d93a7e7b7f874dd281cbf(processScriptRemoteCall)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> d117832c6c2ad667e6a579e0d74d6ecdac37a8a81a25e6168862c4614e0a21e6(processScriptRRefFetchCall)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> 395005eb43a86d29d9d6910a71aedabe2e1e0d8f0c159a20a43035850376190b(processBackwardAutogradReq):::mainFlowStyle
end

subgraph torchcsrcdistributedautogradenginedistenginecpp["torch/csrc/distributed/autograd/engine/dist_engine.cpp"]
395005eb43a86d29d9d6910a71aedabe2e1e0d8f0c159a20a43035850376190b(processBackwardAutogradReq):::mainFlowStyle --> e500f45de6ccbe2e16f2107c96d06e9e777178db49521f1636f3d38d15b0c3c0(executeSendFunctionAsync):::mainFlowStyle
end

subgraph torchcsrcdistributedautogradenginedistenginecpp["torch/csrc/distributed/autograd/engine/dist_engine.cpp"]
e500f45de6ccbe2e16f2107c96d06e9e777178db49521f1636f3d38d15b0c3c0(executeSendFunctionAsync):::mainFlowStyle --> 0bd77ba094d84790b43920190e25c9102c5c085c5f0144152fa41c9889c6fee7(runEngineAndAccumulateGradients):::mainFlowStyle
end

subgraph torchcsrcdistributedautogradenginedistenginecpp["torch/csrc/distributed/autograd/engine/dist_engine.cpp"]
0bd77ba094d84790b43920190e25c9102c5c085c5f0144152fa41c9889c6fee7(runEngineAndAccumulateGradients):::mainFlowStyle --> a80f84ef18a07e7dfb40af610cc475715da206776a64e6c40daea7194b44f2e7(execute_graph_task_until_ready_queue_empty):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
a80f84ef18a07e7dfb40af610cc475715da206776a64e6c40daea7194b44f2e7(execute_graph_task_until_ready_queue_empty):::mainFlowStyle --> c4dbdcd242b5087621fdcb23b2d18943e964d7a5a3f9689912a3acd30c0adb4e(evaluate_function):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
c4dbdcd242b5087621fdcb23b2d18943e964d7a5a3f9689912a3acd30c0adb4e(evaluate_function):::mainFlowStyle --> 2b7323e1c0592760b3bfb3d335424199df959a2a2150a54dd2ecc4f5045f40a1(call_function):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
2b7323e1c0592760b3bfb3d335424199df959a2a2150a54dd2ecc4f5045f40a1(call_function):::mainFlowStyle --> 282eaf412b3f43d49cd2cf71d0a1717ae2a71de65ba1b6ed951d19e6166bbd4b(validate_outputs):::mainFlowStyle
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
      subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
310082d993a7c55b78c5e17623d267c8b5195c68c44ced95532408a33acc6486(processRpcWithErrors):::mainFlowStyle --> 4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> fe13cd6f8c31c181e7e9e2d4abc935cde862bfa7910bb9f725510e051c46bdfe(processCleanupAutogradContextReq)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> accb07d8d1032d75dcbc1a0844da633bd21c1c06196d93a7e7b7f874dd281cbf(processScriptRemoteCall)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> d117832c6c2ad667e6a579e0d74d6ecdac37a8a81a25e6168862c4614e0a21e6(processScriptRRefFetchCall)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
4e17d9f268bec7640a2a0e9ec63f581eb6bcee9f2579633a9dea0adf7de1cc52(processRpc):::mainFlowStyle --> 395005eb43a86d29d9d6910a71aedabe2e1e0d8f0c159a20a43035850376190b(processBackwardAutogradReq):::mainFlowStyle
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
395005eb43a86d29d9d6910a71aedabe2e1e0d8f0c159a20a43035850376190b(processBackwardAutogradReq):::mainFlowStyle --> 4fk2k(...)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
d117832c6c2ad667e6a579e0d74d6ecdac37a8a81a25e6168862c4614e0a21e6(processScriptRRefFetchCall) --> 41309dd630415934c2d2dee7bead65daa6f4936dca5666dfa9d10d5a638721de(retrieveOwnerRRef)
end

subgraph torchcsrcdistributedrpc["torch/csrc/distributed/rpc"]
accb07d8d1032d75dcbc1a0844da633bd21c1c06196d93a7e7b7f874dd281cbf(processScriptRemoteCall) --> af159c25ba3d4b8b4f981cb5a4b970d77760cac2b027ad2ba59f31bd0af6eca0(assignOwnerRRef)
end

subgraph torchcsrcdistributedautogradcontextcontainercpp["torch/csrc/distributed/autograd/context/container.cpp"]
fe13cd6f8c31c181e7e9e2d4abc935cde862bfa7910bb9f725510e051c46bdfe(processCleanupAutogradContextReq) --> b738caa6585a0c924e0bb97971e465f9c33516037f93aa61a4baff4f6f745c32(releaseContextIfPresent)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_impl.cpp" line="253">

---

## Handling RPC with Errors

The function `processRpcWithErrors` is responsible for handling RPC commands and managing any errors that occur during their execution. It attempts to process the RPC command using `processRpc`. If a Python exception is encountered, it captures and handles the error, ensuring that the Python Global Interpreter Lock (GIL) is acquired before manipulating Python objects. For standard exceptions, it handles the error and returns a future with the error information.

```c++
  try {
    return processRpc(rpc, messageType, streams);
  } catch (py::error_already_set& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    auto future = asFuture(handleError(e, messageType, -1));
    // There are request callback impls in Python, where Python
    // exceptions could be thrown. For releasing Python exception
    // py::objects, GIL must be held.
    py::gil_scoped_acquire acquire;
    e.restore(); // Release ownership on py::objects and also restore
                 // Python Error Indicator.
    PyErr_Clear(); // Clear the Python Error Indicator as we has
                   // recorded the exception in the response message.
    return future;
  } catch (std::exception& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    return asFuture(handleError(e, messageType, -1));
  }
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="503">

---

## Processing RPC Commands

The function `processRpc` processes various types of RPC commands based on the `messageType`. It uses a switch statement to delegate the processing to specific functions like `processScriptCall`, `processPythonCall`, `processScriptRemoteCall`, and others. This modular approach allows for handling different RPC command types efficiently.

```c++
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      return processScriptCall(rpc, streams);
    }
    case MessageType::PYTHON_CALL: {
      return processPythonCall(rpc, streams);
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      return processScriptRemoteCall(rpc, streams);
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      return processPythonRemoteCall(rpc, streams);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      return processScriptRRefFetchCall(rpc);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      return processPythonRRefFetchCall(rpc);
    }
    case MessageType::RREF_USER_DELETE: {
      return processRRefUserDelete(rpc);
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="388">

---

### Cleaning Up Autograd Context

The function `processCleanupAutogradContextReq` handles the cleanup of autograd contexts. It releases the context if it still exists, ensuring that any nested RPCs are managed correctly. This is crucial for maintaining the integrity of distributed autograd operations.

```c++
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processCleanupAutogradContextReq(RpcCommandBase& rpc) const {
  auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
  auto cleanupContextId = cleanupContextReq.getContextId();
  // release the context if it still exists on this thread. We need to
  // check if it exists since it may have been deleted by an in-flight
  // RPC. This can create nested RPCs if there are other nodes that get
  // notified to clean up their context.
  DistAutogradContainer::getInstance().releaseContextIfPresent(
      cleanupContextId);
  return asFuture(CleanupAutogradContextResp().toMessage());
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="203">

---

### Processing Script Remote Call

The function `processScriptRemoteCall` processes remote script calls by running the JIT operator and assigning the owner RRef. This ensures that the remote call is executed correctly and the results are properly managed.

```c++
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptRemoteCall(
    RpcCommandBase& rpc,
    const std::vector<c10::Stream>& streams) const {
  auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);

  TORCH_CHECK(
      scriptRemoteCall.hasOp(), "ScriptRemoteCall needs to have an op!");
  auto future = runJitOperator(
      *scriptRemoteCall.op(), scriptRemoteCall.stackRef(), streams);

  return assignOwnerRRef(
      scriptRemoteCall.retRRefId(), scriptRemoteCall.retForkId(), future);
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="234">

---

### Fetching Script RRef

The function `processScriptRRefFetchCall` retrieves the owner RRef for a given script RRef fetch call. It ensures that the future result is correctly handled and returned.

```c++
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processScriptRRefFetchCall(RpcCommandBase& rpc) const {
  auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);

  auto future = retrieveOwnerRRef(srf.rrefId());

  return future->then(
      [](JitFuture& future) {
        return withStorages(ScriptRRefFetchRet({future.value()}).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="217">

---

### Retrieving Owner RRef

The function `retrieveOwnerRRef` retrieves the owner RRef for a given RRef ID. It ensures that the RRef is correctly fetched and the future result is managed appropriately.

```c++
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::retrieveOwnerRRef(
    const RRefId& rrefId) const {
  auto& ctx = RRefContext::getInstance();

  auto rrefFuture = ctx.getOwnerRRef(rrefId);

  at::TypePtr type = rrefFuture->elementType();
  TORCH_INTERNAL_ASSERT(type->kind() == at::RRefType::Kind);
  return rrefFuture->thenAsync(
      [](JitFuture& rrefFuture) {
        c10::intrusive_ptr<OwnerRRef> rref =
            fromRRefInterface(rrefFuture.value().toRRef());
        return rref->getFuture();
      },
      type->cast<at::RRefType>()->getElementType());
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="165">

---

### Assigning Owner RRef

The function `assignOwnerRRef` assigns the owner RRef for a given RRef ID and fork ID. It ensures that the value future is correctly set and any errors are handled appropriately.

```c++
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::assignOwnerRRef(
    const RRefId& rrefId,
    const RRefId& forkId,
    const c10::intrusive_ptr<JitFuture>& valueFuture) const {
  auto& ctx = RRefContext::getInstance();

  c10::intrusive_ptr<OwnerRRef> ownerRRef;
  if (rrefId == forkId) {
    // Creating an owner RRef on self, should already exist in owners map
    ownerRRef =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* forceCreated */ true)
                              ->constValue()
                              .toRRef());
  } else {
    ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, valueFuture->elementType());
    // Caller is a user and callee is the owner, add fork
    //
    // NB: rrefId == forkId is true if and only if calling remote to self.
    // In that case both the caller and the callee will access the
    // OwnerRRef. Hence, on the callee side (here), it should not call
    // addForkOfOwner as it is not a fork. To allow callee to distinguish
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/autograd/context/container.cpp" line="182">

---

### Releasing Context If Present

The function `releaseContextIfPresent` releases the autograd context if it exists. It ensures that any known worker IDs are notified to release their contexts, maintaining the consistency of distributed autograd operations.

```c++
void DistAutogradContainer::releaseContextIfPresent(int64_t context_id) {
  auto& shard = getShard(context_id);
  std::unique_lock<std::mutex> lock(shard.lock);
  auto it = shard.contexts.find(context_id);

  // no-op if the context does not exist on this thread. This could happen if an
  // in-flight RPC has already released the context on this thread.
  if (it == shard.contexts.end()) {
    return;
  }

  auto knownWorkerIds = it->second->getKnownWorkerIds();
  eraseContextIdAndReset(shard, context_id);

  // Unlock since we no longer need the lock.
  lock.unlock();
  sendReleaseContextRpc(knownWorkerIds, context_id);
}
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchcsrcdistributed["torch/csrc/distributed"]
395005eb43a86d29d9d6910a71aedabe2e1e0d8f0c159a20a43035850376190b(processBackwardAutogradReq):::mainFlowStyle --> e500f45de6ccbe2e16f2107c96d06e9e777178db49521f1636f3d38d15b0c3c0(executeSendFunctionAsync):::mainFlowStyle
end

subgraph torchcsrcdistributed["torch/csrc/distributed"]
e500f45de6ccbe2e16f2107c96d06e9e777178db49521f1636f3d38d15b0c3c0(executeSendFunctionAsync):::mainFlowStyle --> 0bd77ba094d84790b43920190e25c9102c5c085c5f0144152fa41c9889c6fee7(runEngineAndAccumulateGradients):::mainFlowStyle
end

subgraph torchcsrcdistributed["torch/csrc/distributed"]
0bd77ba094d84790b43920190e25c9102c5c085c5f0144152fa41c9889c6fee7(runEngineAndAccumulateGradients):::mainFlowStyle --> a80f84ef18a07e7dfb40af610cc475715da206776a64e6c40daea7194b44f2e7(execute_graph_task_until_ready_queue_empty):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
a80f84ef18a07e7dfb40af610cc475715da206776a64e6c40daea7194b44f2e7(execute_graph_task_until_ready_queue_empty):::mainFlowStyle --> c4dbdcd242b5087621fdcb23b2d18943e964d7a5a3f9689912a3acd30c0adb4e(evaluate_function):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
c4dbdcd242b5087621fdcb23b2d18943e964d7a5a3f9689912a3acd30c0adb4e(evaluate_function):::mainFlowStyle --> 2b7323e1c0592760b3bfb3d335424199df959a2a2150a54dd2ecc4f5045f40a1(call_function):::mainFlowStyle
end

subgraph torchcsrcautogradenginecpp["torch/csrc/autograd/engine.cpp"]
2b7323e1c0592760b3bfb3d335424199df959a2a2150a54dd2ecc4f5045f40a1(call_function):::mainFlowStyle --> 282eaf412b3f43d49cd2cf71d0a1717ae2a71de65ba1b6ed951d19e6166bbd4b(validate_outputs):::mainFlowStyle
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="357">

---

## processBackwardAutogradReq

The function `processBackwardAutogradReq` is responsible for handling backward autograd requests in a distributed setting. It retrieves the appropriate autograd context and send function, attaches gradients to the send function, and then executes the autograd graph using the distributed engine. The response is satisfied when the RPCs come back.

```c++
  c10::MultiStreamGuard guard(streams);
  auto& gradientsCall = static_cast<PropagateGradientsReq&>(rpc);
  const auto& autogradMetadata = gradientsCall.getAutogradMetadata();

  // Retrieve the appropriate autograd context.
  auto autogradContext = DistAutogradContainer::getInstance().retrieveContext(
      autogradMetadata.autogradContextId);

  // Lookup the appropriate 'send' function to enqueue.
  std::shared_ptr<SendRpcBackward> sendFunction =
      autogradContext->retrieveSendFunction(autogradMetadata.autogradMessageId);

  // Attach the gradients to the send function.
  sendFunction->setGrads(gradientsCall.getGrads());

  // Now execute the autograd graph using the "distributed engine."
  auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
      autogradContext, sendFunction, gradientsCall.retainGraph());

  // Our response is satisfied when the rpcs come back.
  return execFuture->then(
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="362">

---

### Retrieving Autograd Context

The autograd context is retrieved using the autograd metadata from the gradients call. This context is essential for managing the state and execution of the autograd graph.

```c++
  auto autogradContext = DistAutogradContainer::getInstance().retrieveContext(
      autogradMetadata.autogradContextId);
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/distributed/rpc/request_callback_no_python.cpp" line="373">

---

### Executing Send Function

The send function, which has the gradients attached, is executed asynchronously using the distributed engine. This step is crucial for propagating gradients across different nodes in the distributed setup.

```c++
  auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
      autogradContext, sendFunction, gradientsCall.retainGraph());

```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
