---
title: Overview of the parseIR Function
---
This document provides an overview of the `parseIR` function, which is responsible for initializing the IR parser and starting the parsing process. The document also covers the various steps involved in parsing the graph, including parsing graph inputs, operators, and blocks.

The flow starts with the `parseIR` function, which initializes the IR parser with the provided string, graph, and value map. It then calls the `parse` method to start the parsing process. The `parse` method first parses the graph inputs, then the list of operators, and finally the return statement. It also handles deferred initializations for tensor values and empty containers. The flow continues with parsing individual operators, their inputs, and outputs, and setting their types. Finally, it parses blocks and their inputs, ensuring that all elements are correctly added to the graph.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f8d401eae40cfcd864ddd5a8605e077b490cb38177bd7751a6bd6058f9b51046(parseIR):::mainFlowStyle --> 8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle --> 5a1a23120cdf8d65b73e0e40d2a58c046798ce0d539705f40a87f88001446416(parseGraphInputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle --> f740577e21b0ced574f79fe1e885b2ff29f3534e08293e9388d69d7b404d42a2(parseOperatorsList):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f740577e21b0ced574f79fe1e885b2ff29f3534e08293e9388d69d7b404d42a2(parseOperatorsList):::mainFlowStyle --> 721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> 8eaadbbf8717c317eded3790901f3cb825f07da0587d0821422790249934e95c(parseOperatorInputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> d6fbce150333d0c4166e33f6e8f90bf44e3bb62bd019a4e9e459c13b86f692d0(parseOperatorOutputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> f4d42436fc042ab2a3224e197b4bdfe58ca42284c00a979bffdb90f062c7fa3a(parseBlocks):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f4d42436fc042ab2a3224e197b4bdfe58ca42284c00a979bffdb90f062c7fa3a(parseBlocks):::mainFlowStyle --> 0409e795b7889e0de00baea980042aa7368f818856579f82eddb968dbfecacf0(parseBlock):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
0409e795b7889e0de00baea980042aa7368f818856579f82eddb968dbfecacf0(parseBlock):::mainFlowStyle --> 6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle --> e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle --> fd1f90bcf362f82b4b022ca56f06f05b51a443700dd70da2a975c00428409bea(parseVarWithType):::mainFlowStyle
end

subgraph torchcsrcjitfrontendschematypeparsercpp["torch/csrc/jit/frontend/schema_type_parser.cpp"]
fd1f90bcf362f82b4b022ca56f06f05b51a443700dd70da2a975c00428409bea(parseVarWithType):::mainFlowStyle --> 70a77628f992149fba6b63c64080cdcce52f80c1124d73530e7695c05a15a89e(parseType):::mainFlowStyle
end

subgraph torchcsrcjitfrontendschematypeparsercpp["torch/csrc/jit/frontend/schema_type_parser.cpp"]
70a77628f992149fba6b63c64080cdcce52f80c1124d73530e7695c05a15a89e(parseType):::mainFlowStyle --> d6e6f196fe5682cda364fa9094762d81b85a1a4456f49c690b0f9cdc02a67fb2(parseFakeAndRealType):::mainFlowStyle
end

subgraph torchcsrcjitfrontendschematypeparsercpp["torch/csrc/jit/frontend/schema_type_parser.cpp"]
d6e6f196fe5682cda364fa9094762d81b85a1a4456f49c690b0f9cdc02a67fb2(parseFakeAndRealType):::mainFlowStyle --> fbf927a969366bc53d132ea204128cc6eb1318e48c0a2b3c2a3ff0ad79795395(parseRefinedTensor):::mainFlowStyle
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
      subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f8d401eae40cfcd864ddd5a8605e077b490cb38177bd7751a6bd6058f9b51046(parseIR):::mainFlowStyle --> 8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle --> 5a1a23120cdf8d65b73e0e40d2a58c046798ce0d539705f40a87f88001446416(parseGraphInputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
8884cabded01c299494f8da8dbd748cb12ef9992c41b61290f4ed1bbf3345535(parse):::mainFlowStyle --> f740577e21b0ced574f79fe1e885b2ff29f3534e08293e9388d69d7b404d42a2(parseOperatorsList):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f740577e21b0ced574f79fe1e885b2ff29f3534e08293e9388d69d7b404d42a2(parseOperatorsList):::mainFlowStyle --> 7z2a4(...)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
5a1a23120cdf8d65b73e0e40d2a58c046798ce0d539705f40a87f88001446416(parseGraphInputs) --> e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType)
end

subgraph torchcsrcjittensorexprkernelh["torch/csrc/jit/tensorexpr/kernel.h"]
e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType) --> 9b220fe32cf0749d7be6f263aa93ac62d5f38575dd30ec2f7c1c2f80076c3251(fallback)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="108">

---

## parseIR Function

The `parseIR` function initializes the `IRParser` with the provided string, graph, and value map, and then calls the `parse` method to start the parsing process.

```c++
void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants) {
  torch::jit::IRParser p(str, graph, vmap, parse_tensor_constants);
  p.parse();
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="622">

---

## parse Method

The `parse` method is responsible for parsing the entire graph. It first parses the graph inputs, then the list of operators, and finally the return statement. It also handles deferred initializations for tensor values and empty containers.

```c++
void IRParser::parse() {
  // Parse graph definition, it should look like the following:
  // graphName (input1, input2, ... inputN):
  L.expect(TK_IDENT);
  parseGraphInputs();
  L.expect(':');

  // After the definition we should have a list of statements, parse it:
  parseOperatorsList(g->block());

  // The last statement should be return, which specifies graph outputs
  parseReturnOperator();

  for (Node* n : deferred_tensor_value_initializations_) {
    auto type = n->output()->type()->expect<TensorType>();
    auto tt = n->output()->type()->cast<TensorType>();
    TORCH_INTERNAL_ASSERT(tt, "expected tensor output ", *n);
    auto sizes = tt->sizes().concrete_sizes();
    TORCH_INTERNAL_ASSERT(sizes);
    auto strides = tt->strides().concrete_sizes();
    TORCH_INTERNAL_ASSERT(strides);
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="581">

---

### Parsing Graph Inputs

The `parseGraphInputs` method parses the list of graph inputs, adds them to the graph, and sets their types.

```c++
void IRParser::parseGraphInputs() {
  parseList('(', ',', ')', [&] {
    VarWithType v = parseVarWithType();
    // If the name isn't valid, don't use it
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    vmap[v.name] = g->addInput(uniq_name);
    vmap[v.name]->setType(v.type);
  });
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/ir.h" line="1502">

---

### Setting Types

The `setType` method sets the type of a value. If the type is dynamic, it falls back to a more specific type.

```c
inline Value* Value::setType(TypePtr type) {
  AT_ASSERT(type);
  if (auto dyn = type->castRaw<c10::DynamicType>()) {
    type = dyn->fallback();
  }
  type_ = std::move(type);
  for (Use& use : uses_) {
    use.user->op_ = nullptr;
  }
  return this;
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/tensorexpr/kernel.h" line="158">

---

### Fallback Mechanism

The `fallback` function runs the interpreter state on the provided stack, serving as a fallback mechanism.

```c
  void fallback(Stack& stack) const {
    InterpreterState(code_).run(stack);
  }
  void recompile();
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f740577e21b0ced574f79fe1e885b2ff29f3534e08293e9388d69d7b404d42a2(parseOperatorsList):::mainFlowStyle --> 721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> 8eaadbbf8717c317eded3790901f3cb825f07da0587d0821422790249934e95c(parseOperatorInputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> d6fbce150333d0c4166e33f6e8f90bf44e3bb62bd019a4e9e459c13b86f692d0(parseOperatorOutputs)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
721f44c6c7e5f7921d9f52768dc3b6f4849c627605f6b77b150018aa4f6a33ac(parseOperator):::mainFlowStyle --> f4d42436fc042ab2a3224e197b4bdfe58ca42284c00a979bffdb90f062c7fa3a(parseBlocks):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f4d42436fc042ab2a3224e197b4bdfe58ca42284c00a979bffdb90f062c7fa3a(parseBlocks):::mainFlowStyle --> 7bdru(...)
end

subgraph torchcsrcjittensorexprkernelh["torch/csrc/jit/tensorexpr/kernel.h"]
e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType) --> 9b220fe32cf0749d7be6f263aa93ac62d5f38575dd30ec2f7c1c2f80076c3251(fallback)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
8eaadbbf8717c317eded3790901f3cb825f07da0587d0821422790249934e95c(parseOperatorInputs) --> 24f3883d924844d98adf5956d84674303c0a34ff5c7f16a5782b2b2479b4babb(parseAttrs)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="500">

---

## Parsing the List of Operators

The function `parseOperatorsList` is responsible for parsing a list of operators within a block. It expects the list to be indented and delimited by either `TK_NEWLINE`, `TK_RETURN`, or `TK_ARROW`. The function iterates through the list and calls `parseOperator` for each operator found.

```c++
  L.expect(TK_INDENT);
  while (L.cur().kind != TK_ARROW && L.cur().kind != TK_RETURN) {
    parseOperator(b);
  }
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="522">

---

## Parsing Individual Operators

The function `parseOperator` parses an individual operator statement. It handles the parsing of the operator's outputs, name, attributes, and inputs. It also registers the outputs and inserts the new node into the block. If the operator has nested blocks, it calls `parseBlocks` to handle them.

```c++
  // Parse lefthand side.
  std::vector<VarWithType> outs;
  parseOperatorOutputs(&outs);

  // Parse the name and create the corresponding node in the graph.
  auto source_range = L.cur().range;
  std::string name = parseOperatorName();
  Node* n = g->create(Symbol::fromQualString(name), {}, outs.size())
                ->setSourceRange(source_range);

  // Parse attributes and inputs.
  parseOperatorInputs(n);

  const FunctionSchema* schema = n->maybeSchema();

  // Register outputs.
  unsigned idx = 0;
  for (const VarWithType& v : outs) {
    vmap[v.name] = n->outputs()[idx];
    if (schema && !schema->is_varret()) {
      TORCH_CHECK(
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="437">

---

### Parsing Operator Inputs

The function `parseOperatorInputs` parses the inputs of an operator. If the current token is '\[', it calls `parseAttrs` to parse the attributes. It then parses a list of inputs enclosed in parentheses and adds each input to the node.

```c++
void IRParser::parseOperatorInputs(Node* n) {
  if (L.cur().kind == '[') {
    parseAttrs(n);
  }
  parseList('(', ',', ')', [&] {
    std::string var_name = parseVar();
    n->addInput(findValueInVMap(var_name));
  });
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="162">

---

### Parsing Operator Outputs

The function `parseOperatorOutputs` parses the outputs of an operator. If the current token is '%', it parses a list of outputs and expects an '=' token to follow.

```c++
void IRParser::parseOperatorOutputs(std::vector<VarWithType>* outs) {
  if (L.cur().kind != '%') {
    return;
  }
  parseList(TK_NOTHING, ',', TK_NOTHING, [&] {
    outs->push_back(parseVarWithType(true));
  });
  L.expect('=');
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="433">

---

### Parsing Attributes

The function `parseAttrs` parses the attributes of an operator. It parses a list of attributes enclosed in square brackets and calls `parseAttr` for each attribute.

```c++
void IRParser::parseAttrs(Node* n) {
  parseList('[', ',', ']', [&] { parseAttr(n); });
}
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchcsrcjitir["torch/csrc/jit/ir"]
f4d42436fc042ab2a3224e197b4bdfe58ca42284c00a979bffdb90f062c7fa3a(parseBlocks):::mainFlowStyle --> 0409e795b7889e0de00baea980042aa7368f818856579f82eddb968dbfecacf0(parseBlock):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
0409e795b7889e0de00baea980042aa7368f818856579f82eddb968dbfecacf0(parseBlock):::mainFlowStyle --> 6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle --> e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType)
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
6641a9b0870a4ca12733bb85bcf1a1d7213fc6366aac02e833289a1be17cb2f0(parseBlockInputs):::mainFlowStyle --> fd1f90bcf362f82b4b022ca56f06f05b51a443700dd70da2a975c00428409bea(parseVarWithType):::mainFlowStyle
end

subgraph torchcsrcjitir["torch/csrc/jit/ir"]
fd1f90bcf362f82b4b022ca56f06f05b51a443700dd70da2a975c00428409bea(parseVarWithType):::mainFlowStyle --> kv7w3(...)
end

subgraph torchcsrcjittensorexprkernelh["torch/csrc/jit/tensorexpr/kernel.h"]
e313a530025028a575514f8176a33b30f9300314b506395d99a9640d19b8c1e8(setType) --> 9b220fe32cf0749d7be6f263aa93ac62d5f38575dd30ec2f7c1c2f80076c3251(fallback)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="447">

---

## Parsing Blocks

The function `parseBlocks` is responsible for parsing multiple blocks within a parent node. It expects an indentation to start and continues to parse individual blocks using the `parseBlock` function until it encounters a dedentation.

```c++
void IRParser::parseBlocks(Node* parentNode) {
  L.expect(TK_INDENT);
  while (L.cur().kind != TK_DEDENT) {
    parseBlock(parentNode);
  }
  L.expect(TK_DEDENT);
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="475">

---

## Parsing a Single Block

The function `parseBlock` handles the parsing of a single block. It adds a new block to the parent node, parses the block's inputs using `parseBlockInputs`, expects a colon to denote the start of operations, parses the list of operations, and finally parses the block's outputs.

```c++
/** \brief Parse a block.
 *
 * It should look like the following:
 * blockName(input1, input2, input3, ...):
 *   op1
 *   op2
 *   ...
 *   opN
 *   -> (output1, output2, output3, ...)
 */
void IRParser::parseBlock(Node* parentNode) {
  Block* b = parentNode->addBlock();
  L.expect(TK_IDENT).text(); // Block name is not used anywhere.
  parseBlockInputs(b);
  L.expect(':');
  parseOperatorsList(b);
  parseBlockOutputs(b);
}
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="455">

---

### Parsing Block Inputs

The function `parseBlockInputs` parses the inputs of a block. It reads a list of variables with types, validates their names, adds them as inputs to the block, and sets their types accordingly.

```c++
void IRParser::parseBlockInputs(Block* b) {
  parseList('(', ',', ')', [&] {
    VarWithType v = parseVarWithType();
    // If the name isn't valid, don't use it
    std::string uniq_name = Value::isValidName(v.name) ? v.name : "";
    vmap[v.name] = b->addInput(uniq_name);
    vmap[v.name]->setType(v.type);
  });
}
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchcsrcjit["torch/csrc/jit"]
fd1f90bcf362f82b4b022ca56f06f05b51a443700dd70da2a975c00428409bea(parseVarWithType):::mainFlowStyle --> 70a77628f992149fba6b63c64080cdcce52f80c1124d73530e7695c05a15a89e(parseType):::mainFlowStyle
end

subgraph torchcsrcjit["torch/csrc/jit"]
70a77628f992149fba6b63c64080cdcce52f80c1124d73530e7695c05a15a89e(parseType):::mainFlowStyle --> d6e6f196fe5682cda364fa9094762d81b85a1a4456f49c690b0f9cdc02a67fb2(parseFakeAndRealType):::mainFlowStyle
end

subgraph torchcsrcjit["torch/csrc/jit"]
d6e6f196fe5682cda364fa9094762d81b85a1a4456f49c690b0f9cdc02a67fb2(parseFakeAndRealType):::mainFlowStyle --> fbf927a969366bc53d132ea204128cc6eb1318e48c0a2b3c2a3ff0ad79795395(parseRefinedTensor):::mainFlowStyle
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/csrc/jit/ir/irparser.cpp" line="128">

---

## Parsing Variable with Type

The function `parseVarWithType` is responsible for parsing a variable and its associated type. It first parses the variable name using `parseVar`. If the `allow_optional` flag is set, the type is set to `nullptr`; otherwise, it defaults to `TensorType::get()`. If a colon is encountered, it indicates the presence of a type alias, which is then parsed using `type_parser.parseType()`. The parsed type is assigned to the variable.

```c++
  if (allow_optional) {
    r.type = nullptr;
  } else {
    r.type = TensorType::get();
  }
  if (L.nextIf(':')) {
    auto type_alias = type_parser.parseType();
    AT_ASSERTM(!type_alias.second, "Parsing IR with Alias Info not handled");
    r.type = type_alias.first;
  }
  return r;
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/frontend/schema_type_parser.cpp" line="358">

---

## Parsing Type

The function `parseType` calls `parseFakeAndRealType` to parse both the fake and real types. It returns a pair consisting of the parsed type and any associated alias information.

```c++
  auto r = parseFakeAndRealType();
  return std::make_pair(std::move(std::get<0>(r)), std::move(std::get<2>(r)));
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/frontend/schema_type_parser.cpp" line="362">

---

## Parsing Fake and Real Type

The function `parseFakeAndRealType` handles the parsing of various types, including tuples, futures, awaits, RRefs, tensors, dictionaries, unions, and custom classes. It also manages alias information and handles optional types and lists. The function ensures that the parsed types are correctly associated with their aliases and other attributes like device and requires_grad.

```c++
std::tuple</*fake*/ TypePtr, /*real*/ TypePtr, std::optional<AliasInfo>>
SchemaTypeParser::parseFakeAndRealType() {
  TypePtr fake_value;
  TypePtr real_value;
  std::optional<AliasInfo> alias_info;
  // Tuple type
  if (L.cur().kind == '(') {
    std::vector<TypePtr> types;
    parseList('(', ',', ')', [&] {
      auto r = parseType();
      types.push_back(std::move(r.first));
      if (alias_info && r.second) {
        alias_info->addContainedType(std::move(*r.second));
      }
    });
    fake_value = real_value =
        c10::TypeFactory::create<TupleType>(std::move(types));
  } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
    L.next(); // Future
    L.expect('(');
    auto p = parseType();
```

---

</SwmSnippet>

<SwmSnippet path="/torch/csrc/jit/frontend/schema_type_parser.cpp" line="229">

---

## Parsing Refined Tensor

The function `parseRefinedTensor` parses a refined tensor type, including its data type, dimensions, strides, device, and requires_grad attributes. It supports various tensor specifications, such as known and unknown ranks, and ensures that the parsed tensor type is correctly constructed with all its attributes.

```c++
TypePtr SchemaTypeParser::parseRefinedTensor() {
  auto maybe_dtype = parseTensorDType(L.expect(TK_IDENT).text());
  AT_ASSERT(maybe_dtype);
  at::ScalarType dtype = *maybe_dtype;
  TypePtr ptr;
  L.expect('(');
  TypePtr tensor_type;
  std::optional<c10::Device> device;
  std::optional<bool> requires_grad;
  // Parse a type with either no ranks, known ranks with sizes, ranks with
  // unknown sizes, a mix of ranks with known and unknown sizes, or ranks with
  // known sizes and strides. The type might also have requires_grad and/or
  // device option. Examples of types we're handling here:
  //   Long(10, 8, 6, strides=[48, 6, 1], requires_grad=0, device=cuda:1)
  //   Float(10, *, 20, device=cuda:1)
  //   Float(requires_grad=1)
  std::vector<std::optional<int64_t>> dims;
  bool seen_strides = false;
  std::vector<int64_t> strides;
  parseList(TK_NOTHING, ',', ')', [&] {
    // Extra handling for options like 'device' and 'requires_grad'
```

---

</SwmSnippet>

# Where is this flow used?

This flow is used multiple times in the codebase as represented in the following diagram:

(Note - these are only some of the entry points of this flow)

```mermaid
graph TD;
      subgraph torchcsrcjit["torch/csrc/jit"]
2571659e34f457caf4046d68664653c4ec8faf0274fb8858829b6a586b2e2d59(load_jit_module_from_file):::rootsStyle --> 1d1e06558a0856dfc5b7765130ef34d14851c7900865897edfd3691eace94b29(parse_and_initialize_jit_module)
end

subgraph torchcsrcjit["torch/csrc/jit"]
1d1e06558a0856dfc5b7765130ef34d14851c7900865897edfd3691eace94b29(parse_and_initialize_jit_module) --> 5e1ccb3847488fc3a2f625a4629d3f145afee9cd5b29047717e7dca0631921d7(jitModuleFromSourceAndConstants)
end

subgraph torchcsrcjit["torch/csrc/jit"]
5e1ccb3847488fc3a2f625a4629d3f145afee9cd5b29047717e7dca0631921d7(jitModuleFromSourceAndConstants) --> cb8521ec0d4f8bcfef2f7f991ddee049eb42b0c46e6dbd01790421903ae51a9b(rewriteQuantizedConvForBC)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
cb8521ec0d4f8bcfef2f7f991ddee049eb42b0c46e6dbd01790421903ae51a9b(rewriteQuantizedConvForBC) --> 604c929cfbbc3d582c2f785a2a705e5f052b8f7f7522011718f1a5a3d8f18e95(runOnModule)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
604c929cfbbc3d582c2f785a2a705e5f052b8f7f7522011718f1a5a3d8f18e95(runOnModule) --> 52a1401de3272a7bded15a38883ca07707144ddb413c3ee27ca7ce8490374940(runOnGraph)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
52a1401de3272a7bded15a38883ca07707144ddb413c3ee27ca7ce8490374940(runOnGraph) --> 70f76c33383db70965a9dcf34f354ebcf94d387b05498ad216f822f8686fae09(rewriteSinglePatternOnGraph)
end

subgraph torchcsrcjit["torch/csrc/jit"]
70f76c33383db70965a9dcf34f354ebcf94d387b05498ad216f822f8686fae09(rewriteSinglePatternOnGraph) --> f8d401eae40cfcd864ddd5a8605e077b490cb38177bd7751a6bd6058f9b51046(parseIR):::mainFlowStyle
end

subgraph torchcsrcjit["torch/csrc/jit"]
d8204ae8148db11c778b2aaa73ff7171ad689a9ce721972302b4eee4fda424c6(load_jit_module_from_stream):::rootsStyle --> 1d1e06558a0856dfc5b7765130ef34d14851c7900865897edfd3691eace94b29(parse_and_initialize_jit_module)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
1f5b800471b5619d3ecd53a18eb527e0e2f3a99506d6d7a58d17011a43bf99f4(fuseFrozenConvAddReluImpl):::rootsStyle --> 52a1401de3272a7bded15a38883ca07707144ddb413c3ee27ca7ce8490374940(runOnGraph)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
30600bc672f62e415b713a0f0652f5d8332596a760460e73f343103be9032aa3(RemoveRedundantQuantizationOps):::rootsStyle --> 52a1401de3272a7bded15a38883ca07707144ddb413c3ee27ca7ce8490374940(runOnGraph)
end

subgraph torchcsrcjitpasses["torch/csrc/jit/passes"]
a093298eb2e6266fd76621fc99192eddc2149686381a35fe529bcae5c5c3bbe6(RemoveRedundantDequantize):::rootsStyle --> 52a1401de3272a7bded15a38883ca07707144ddb413c3ee27ca7ce8490374940(runOnGraph)
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
