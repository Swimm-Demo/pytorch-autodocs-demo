---
title: Overview of Passes in FX
---
# Overview of Passes in FX

Passes are used to transform or analyze the computational graph in PyTorch's FX module. They provide a mechanism to modify or inspect the graph, ensuring that various transformations and analyses can be applied systematically.

## PassBase Class

The `PassBase` class provides a base interface for implementing passes. It requires the implementation of the `call` function, which defines the pass to be run on the graph module. Optional `requires` and `ensures` methods can be implemented to check preconditions and postconditions of the graph module.

<SwmSnippet path="/torch/fx/passes/infra/pass_base.py" line="23">

---

The `PassBase` class defines the structure for implementing passes. The `__call__` method runs the precondition check, the pass itself, and the postcondition check.

```python
class PassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> Optional[PassResult]:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        return res
```

---

</SwmSnippet>

## PassManager Class

The `PassManager` class is responsible for collecting passes and constraints, defining the pass schedule, and managing pass execution. It ensures that all constraints between passes are respected and provides methods to add passes and constraints, validate them, and execute the passes on a given graph module.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="147">

---

The `PassManager` class manages the scheduling and execution of multiple passes, ensuring that constraints between passes are respected.

```python
class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
        suppress_check_failures (bool): Whether to raise errors when running
            checks
    """
```

---

</SwmSnippet>

## Adding a Pass

The `add_pass` function in the `PassManager` class allows you to add a pass to the current list of passes. This function appends the pass to the list and marks the pass schedule as not validated.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="189">

---

The `add_pass` function appends a pass to the list and marks the pass schedule as not validated.

```python
    def add_pass(self, _pass: Callable):
        """
        Adds a pass into the current list of passes.
        """
        self.passes.append(_pass)
        self._validated = False
```

---

</SwmSnippet>

## Main Functions of Passes

The main functions of passes include `pass_result_wrapper`, `_topological_sort_passes`, `this_before_that_pass_constraint`, `validate_constraints`, `__call__`, and `solve_constraints`.

### pass_result_wrapper

The `pass_result_wrapper` function is a wrapper for passes that do not return a `PassResult`. It ensures that the wrapped function returns a `PassResult` containing the modified object and a flag indicating whether the object was modified.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="20">

---

The `pass_result_wrapper` function ensures that the wrapped function returns a `PassResult`.

```python
def pass_result_wrapper(fn: Callable) -> Callable:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        return None

    @wraps(fn)
    def wrapped_fn(gm):
        res = fn(gm)
        if res is None:
            return PassResult(gm, True)
        if isinstance(res, PassResult):
```

---

</SwmSnippet>

### \_topological_sort_passes

The `_topological_sort_passes` function sorts a list of passes based on given constraints. It returns a sorted list of callables and checks for circular dependencies.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="63">

---

The `_topological_sort_passes` function sorts passes based on constraints and checks for circular dependencies.

```python
def _topological_sort_passes(
    passes: List[Callable], constraints: List[Callable]
) -> List[Callable]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    if len(constraints) == 0:
        return passes

    # Contruct a graph mapping nodes to a list of their users
    graph: Dict[Callable, List[Callable]] = {p : [] for p in passes}
    indegree_map: Dict[Callable, int] = dict.fromkeys(passes, 0)
    candidates: Queue = Queue()
    for a in passes:
        for b in passes:
```

---

</SwmSnippet>

### this_before_that_pass_constraint

The `this_before_that_pass_constraint` function defines a partial order where one pass must occur before another. It returns a callable that enforces this constraint.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="118">

---

The `this_before_that_pass_constraint` function enforces a partial order between passes.

````python
def this_before_that_pass_constraint(this: Callable, that: Callable) -> Callable:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [pass_b, pass_a]

    constraints = [
        this_before_that_pass_constraint(pass_a, pass_b)
    ]
    ```

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """
````

---

</SwmSnippet>

### validate_constraints

The `validate_constraints` function validates that the current pass schedule is valid according to all constraints. It raises an error if any constraint is violated.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="203">

---

The `validate_constraints` function checks if the current pass schedule is valid according to all constraints.

```python
    def validate_constraints(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True
```

---

</SwmSnippet>

### **call**

The `__call__` function runs the list of passes on a given module. It ensures that the passes are run in the correct order and checks the module after each pass if specified.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="242">

---

The `__call__` function runs the passes on a given module in the correct order.

```python
    def __call__(self, module: nn.Module) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
        # Order the passes based on the constraints
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self.check(module)

        # Run the set of passes `steps` number of times or until the graph stops
        # changing
        overall_modified = False
        for _ in range(self.steps):
            modified = False
```

---

</SwmSnippet>

### solve_constraints

The `solve_constraints` function finds a valid traversal order based on the given constraints and orders the passes accordingly. It raises an error if a circular dependency exists.

<SwmSnippet path="/torch/fx/passes/infra/pass_manager.py" line="214">

---

The `solve_constraints` function orders the passes based on constraints and raises an error if a circular dependency exists.

```python
    def solve_constraints(self):
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.

        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
        self.passes = _topological_sort_passes(self.passes, self.constraints)
        self._validated = True
```

---

</SwmSnippet>

&nbsp;

*This is an auto-generated document by Swimm AI 🌊 and has not yet been verified by a human*

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBcHl0b3JjaC1hdXRvZG9jcy1kZW1vJTNBJTNBU3dpbW0tRGVtbw==" repo-name="pytorch-autodocs-demo"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
