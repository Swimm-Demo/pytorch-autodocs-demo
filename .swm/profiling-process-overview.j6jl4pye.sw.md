---
title: Profiling Process Overview
---
This document outlines the profiling process, detailing each step from initialization to retrieving results. The profiling process ensures that no concurrent runs are supported, starts the profiling asynchronously, waits for it to be in the running state, and handles the results after the profiling is stopped.

The profiling process starts by ensuring no other profiling runs are happening simultaneously. It then begins the profiling asynchronously and waits until the profiling is running. Once the profiling is complete, it stops the profiling and collects the results. If any errors occur during this process, they are logged and handled appropriately.

Here is a high level diagram of the flow, showing only the most important functions:

```mermaid
graph TD;
      b7316efd8dd045e2fdd832592c8d8f6342390c3da2816e00b1c4ed03f897dc91(profile):::mainFlowStyle --> 45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle

45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 50d1c38fe02886979eb950c9f9803e8ca00563054ba553eec2c0e332742d73ac(_wait_for_running)

45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 752a4117ca365bd0de2ed3e1a67850b895f3a29e5b2f522538a911bbe8dc749a(_run_async)

45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle

566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle --> 692b63c29ce13f90ce68ff29d4a23b8b951f7350426a6a66505ce69badc54c25(_stop_run)

566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle --> 2173f7e4a53b29c1e75f8c9ddfb77e49f2fb6731351def4afc0ea1163341c30a(_get_results):::mainFlowStyle

2173f7e4a53b29c1e75f8c9ddfb77e49f2fb6731351def4afc0ea1163341c30a(_get_results):::mainFlowStyle --> 836dfa01c863d5b2c4657af573287c9e0f368fbb19a63a3a91b97ae92c305aaf(run):::mainFlowStyle

836dfa01c863d5b2c4657af573287c9e0f368fbb19a63a3a91b97ae92c305aaf(run):::mainFlowStyle --> 52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle

52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle --> fcb45b8524fa47ab2f9646e7465821b1529771cf118d1f70dbefb53ff0c1d44a(_update_basic)

52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle --> 3ec2c7b5acedc1cea8dd6ad5bf173f77d8b70a4f95cf678957502e4e551d8e19(_update_ortho):::mainFlowStyle


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
      subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
b7316efd8dd045e2fdd832592c8d8f6342390c3da2816e00b1c4ed03f897dc91(profile):::mainFlowStyle --> 45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 50d1c38fe02886979eb950c9f9803e8ca00563054ba553eec2c0e332742d73ac(_wait_for_running)
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 752a4117ca365bd0de2ed3e1a67850b895f3a29e5b2f522538a911bbe8dc749a(_run_async)
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
45393de97b0a56bd8294f35d28900fec83bd7ad5dddea66b5e9c897ddbe501e9(_start_strobelight):::mainFlowStyle --> 566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle --> 5honq(...)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="253">

---

## Profile Initialization

The `profile` function initializes the profiling process by acquiring a lock to ensure no concurrent runs are supported. If the lock is successfully acquired, it proceeds to start the strobelight profiling.

```python
    def profile(self, work_function: Any, *args: Any, **kwargs: Any) -> Any:
        self.current_run_id = None
        self.profile_result = None

        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            if not locked:
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")

                logger.warning("concurrent runs not supported")
                return work_function(*args, **kwargs)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="236">

---

## Starting Strobelight

The `_start_strobelight` function attempts to start the strobelight profiling asynchronously. It logs the run ID and waits for the profiling to be in the running state.

```python
    # Return true if strobelight started and is running. Never throw.
    def _start_strobelight(self) -> bool:
        strobelight_started = False
        try:
            self._run_async()
            strobelight_started = True
            logger.info("strobelight run id is: %s", self.current_run_id)
            self._wait_for_running()
            logger.info("strobelight profiling running")
            return True
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="93">

---

### Running Asynchronously

The `_run_async` function constructs and executes the command to start the strobelight profiling in asynchronous mode. It captures the output to extract the run ID.

```python
    def _run_async(self) -> None:
        processId = os.getpid()
        namespace = _pid_namespace(processId)
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            f"{int(self.sample_each)}",
            "--duration-ms",
            f"{int(self.max_profile_duration_sec * 1000)}",
            "--pid",
            f"{namespace}:{processId}",
        ]

        if self.sample_tags:
            command.append("--sample-tags")
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="134">

---

### Waiting for Running State

The `_wait_for_running` function checks the status of the profiling process. It repeatedly queries the status until it confirms that the profiling is running or encounters an error.

```python
    def _wait_for_running(self, counter: int = 0) -> None:
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            if current_status == "RUNNING":
                return
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="276">

---

## Handling Profiling Results

After the work function is executed, the `profile` function stops the strobelight profiling and releases the lock. It handles any exceptions that occur during the profiling process.

```python
            try:
                logger.debug("collection started")
                start = timer()
                result = work_function(*args, **kwargs)
                end = timer()
                total_time = end - start  # Time in seconds, e.g. 5.38091952400282
                logger.info("work function took %s seconds", total_time)
                self._stop_strobelight_no_throw(collect_results=True)
                StrobelightCLIFunctionProfiler._lock.release()
                return result
            except Exception as error:
                logger.warning("work function throw exception", exc_info=True)
                self._stop_strobelight_no_throw(collect_results=False)
                StrobelightCLIFunctionProfiler._lock.release()
                raise error
```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle --> 692b63c29ce13f90ce68ff29d4a23b8b951f7350426a6a66505ce69badc54c25(_stop_run)
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
566dae10333aa40af73fa029c4aca67e67ed68f6b26b9ff584237e0d062056d3(_stop_strobelight_no_throw):::mainFlowStyle --> 2173f7e4a53b29c1e75f8c9ddfb77e49f2fb6731351def4afc0ea1163341c30a(_get_results):::mainFlowStyle
end

subgraph torchstrobelightclifunctionprofilerpy["torch/_strobelight/cli_function_profiler.py"]
2173f7e4a53b29c1e75f8c9ddfb77e49f2fb6731351def4afc0ea1163341c30a(_get_results):::mainFlowStyle --> gu72k(...)
end


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="218">

---

## \_stop_strobelight_no_throw

The function `_stop_strobelight_no_throw` is responsible for stopping the strobelight profiling process. It first attempts to stop the run by calling the `_stop_run` function. If the `collect_results` flag is set to true, it proceeds to collect the results by calling the `_get_results` function. Any exceptions encountered during this process are caught and logged as warnings.

```python
    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        try:
            # call stop run
            self._stop_run()
            logger.info("strobelight profiling stopped")

            logger.debug("collection stopped")

            if not collect_results:
                return

            self._get_results()
        except Exception as error:
            logger.warning("error during stop_strobelight", exc_info=True)
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="164">

---

### \_stop_run

The function `_stop_run` sends a command to stop the current profiling run using the `strobeclient` tool. It constructs the command with the current run ID and executes it using `subprocess.run`. The function checks the command's output and raises an error if the profiling stop was not successful. This function ensures that the profiling process is properly terminated before any results are collected.

```python
    def _stop_run(self) -> None:
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Success!"):
                return
            else:
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

```

---

</SwmSnippet>

Now, lets zoom into this section of the flow:

```mermaid
graph TD;
      2173f7e4a53b29c1e75f8c9ddfb77e49f2fb6731351def4afc0ea1163341c30a(_get_results):::mainFlowStyle --> 836dfa01c863d5b2c4657af573287c9e0f368fbb19a63a3a91b97ae92c305aaf(run):::mainFlowStyle

836dfa01c863d5b2c4657af573287c9e0f368fbb19a63a3a91b97ae92c305aaf(run):::mainFlowStyle --> 52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle

52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle --> fcb45b8524fa47ab2f9646e7465821b1529771cf118d1f70dbefb53ff0c1d44a(_update_basic)

52dd951c8c8d7d3b739de7d0bbe24dc962cb8a3659f026f6cef28044fa30dbc9(update):::mainFlowStyle --> 3ec2c7b5acedc1cea8dd6ad5bf173f77d8b70a4f95cf678957502e4e551d8e19(_update_ortho):::mainFlowStyle


      classDef mainFlowStyle color:#000000,fill:#7CB9F4
classDef rootsStyle color:#000000,fill:#00FFF4
classDef Style1 color:#000000,fill:#00FFAA
classDef Style2 color:#000000,fill:#FFFF00
classDef Style3 color:#000000,fill:#AA7CB9
```

<SwmSnippet path="/torch/_strobelight/cli_function_profiler.py" line="188">

---

## Retrieving Profiling Results

The `_get_results` function is responsible for retrieving the profiling results of a run. It constructs a command to query the status of the profiling run using `strobeclient`. If the run is still processing, it waits and retries. Once the run is complete, it parses the output to extract relevant profiling data and logs it.

```python
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Profile run status: PROCESSING"):
                time.sleep(10)
                self._get_results()
                return
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_lobpcg.py" line="817">

---

## Running LOBPCG Iterations

The `run` function initiates the LOBPCG iterations. It first calls the `update` function to set up iteration variables. If a tracker is present, it calls the tracker before and after the iteration loop. The loop continues to call `update` until the stopping condition is met.

```python
    def run(self):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        self.update()

        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()

        while not self.stop_iteration():
            self.update()

            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_lobpcg.py" line="747">

---

## Updating Iteration Variables

The `update` function sets and updates the iteration variables. It normalizes the input matrices and sets initial values for iteration counters. Depending on the method specified, it calls either `_update_basic` or `_update_ortho` to perform the specific update logic.

```python
    def update(self):
        """Set and update iteration variables."""
        if self.ivars["istep"] == 0:
            X_norm = float(torch.norm(self.X))
            iX_norm = X_norm**-1
            A_norm = float(torch.norm(_utils.matmul(self.A, self.X))) * iX_norm
            B_norm = float(torch.norm(_utils.matmul(self.B, self.X))) * iX_norm
            self.fvars["X_norm"] = X_norm
            self.fvars["A_norm"] = A_norm
            self.fvars["B_norm"] = B_norm
            self.ivars["iterations_left"] = self.iparams["niter"]
            self.ivars["converged_count"] = 0
            self.ivars["converged_end"] = 0

        if self.method == "ortho":
            self._update_ortho()
        else:
            self._update_basic()

        self.ivars["iterations_left"] = self.ivars["iterations_left"] - 1
        self.ivars["istep"] = self.ivars["istep"] + 1
```

---

</SwmSnippet>

<SwmSnippet path="/torch/_lobpcg.py" line="847">

---

### Basic Update Logic

The `_update_basic` function handles the update logic when the method is set to 'basic'. It performs the Rayleigh-Ritz procedure to update the eigenvalues and eigenvectors, checks for convergence, and updates the residuals and iteration variables accordingly.

```python
    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X

```

---

</SwmSnippet>

<SwmSnippet path="/torch/_lobpcg.py" line="890">

---

### Orthogonal Update Logic

The `_update_ortho` function handles the update logic when the method is set to 'ortho'. It performs the Rayleigh-Ritz procedure with orthogonalization to update the eigenvalues and eigenvectors, checks for convergence, and updates the residuals and iteration variables accordingly.

```python
    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
```

---

</SwmSnippet>

# Where is this flow used?

This flow is used multiple times in the codebase as represented in the following diagram:

(Note - these are only some of the entry points of this flow)

```mermaid
graph TD;
      subgraph torchinductor["torch/_inductor"]
e60dd21554cef837462f4efb6b6ea804e556a200dc4da10a2d0e442f279f9ffe(run):::rootsStyle --> f5c355285f05692ed594c6614440a4caba1c40b75b1675704b503b8fa2d77678(coordinate_descent_tuning)
end

subgraph torchinductor["torch/_inductor"]
f5c355285f05692ed594c6614440a4caba1c40b75b1675704b503b8fa2d77678(coordinate_descent_tuning) --> a3d0a5983f094e0a5f373f398fa57bfeda7e1c53fd1b2b31f780a4dd7230ccd2(bench)
end

subgraph torchinductor["torch/_inductor"]
a3d0a5983f094e0a5f373f398fa57bfeda7e1c53fd1b2b31f780a4dd7230ccd2(bench) --> 9a1f93a790c5fb4c42dc7f8094dbaf0cd069d905cc0b125470645167e229b7d2(do_bench_using_profiling)
end

subgraph torchstrobelight["torch/_strobelight"]
9a1f93a790c5fb4c42dc7f8094dbaf0cd069d905cc0b125470645167e229b7d2(do_bench_using_profiling) --> b7316efd8dd045e2fdd832592c8d8f6342390c3da2816e00b1c4ed03f897dc91(profile):::mainFlowStyle
end

subgraph torchdynamobackends["torch/_dynamo/backends"]
21c9681698a31e0a38af130c36b97ad8b7fa65b2f81092a65f276a764ab4c1d1(tvm):::rootsStyle --> 1db1e52db1a3321886aac76d82b1aa2757dcc35ac09c4dc23d46625c8b9be6c0(run)
end

subgraph torchdynamoconvertframepy["torch/_dynamo/convert_frame.py"]
1db1e52db1a3321886aac76d82b1aa2757dcc35ac09c4dc23d46625c8b9be6c0(run) --> 97cb2818a0d7afd41a22dbf90d243a8329e9c794c4c241a8a49b21d9f4504da3(replay)
end

subgraph torchdynamoconvertframepy["torch/_dynamo/convert_frame.py"]
97cb2818a0d7afd41a22dbf90d243a8329e9c794c4c241a8a49b21d9f4504da3(replay) --> bf9d359eb3b1abadd0eb5f8ce8200c1fea4cb5f56120ec2fb67db8332119f0e8(_compile)
end

subgraph torchutilsinternalpy["torch/_utils_internal.py"]
bf9d359eb3b1abadd0eb5f8ce8200c1fea4cb5f56120ec2fb67db8332119f0e8(_compile) --> 4478714677c3f3139b733bd9ec5e99941022b58244475dcf8fc235f430445d5c(compile_time_strobelight_meta)
end

subgraph torchstrobelight["torch/_strobelight"]
4478714677c3f3139b733bd9ec5e99941022b58244475dcf8fc235f430445d5c(compile_time_strobelight_meta) --> 680d1b29416b54596dcb8a5aea2aef4a989d20e73a726d54ee2fa60e3604a404(profile_compile_time)
end

subgraph torchstrobelight["torch/_strobelight"]
680d1b29416b54596dcb8a5aea2aef4a989d20e73a726d54ee2fa60e3604a404(profile_compile_time) --> b7316efd8dd045e2fdd832592c8d8f6342390c3da2816e00b1c4ed03f897dc91(profile):::mainFlowStyle
end

subgraph torchdynamobackends["torch/_dynamo/backends"]
937833df337c443f4ef4c0e13e6738b191455206233c3789689dbcc651820d73(exec_tvm):::rootsStyle --> 1db1e52db1a3321886aac76d82b1aa2757dcc35ac09c4dc23d46625c8b9be6c0(run)
end

subgraph torchdynamobackends["torch/_dynamo/backends"]
3406cb65e615b6f7b327d680190bf2ca057ca8a3c52d51b781d4398535e756c6(compile_fn):::rootsStyle --> 1db1e52db1a3321886aac76d82b1aa2757dcc35ac09c4dc23d46625c8b9be6c0(run)
end

subgraph torchdynamobackends["torch/_dynamo/backends"]
356c2e3b9aa8d609c3283b0721451e81f57a0eaa4d8f1e60fd75694ef1aec032(__call__):::rootsStyle --> 0c04ac0482af24e21109a648a1c93694686c549f26e827f9f635d5d33ccf068b(cudagraphs)
end

subgraph torchinductor["torch/_inductor"]
0c04ac0482af24e21109a648a1c93694686c549f26e827f9f635d5d33ccf068b(cudagraphs) --> 6122bdb06285516ac2cd429df91bb625577e92c7b559b50e12fcdb77d473d6ef(cudagraphify_impl)
end

subgraph torchdynamoconvertframepy["torch/_dynamo/convert_frame.py"]
6122bdb06285516ac2cd429df91bb625577e92c7b559b50e12fcdb77d473d6ef(cudagraphify_impl) --> 97cb2818a0d7afd41a22dbf90d243a8329e9c794c4c241a8a49b21d9f4504da3(replay)
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
