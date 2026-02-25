# Full-suite discovery: 1:1 mapping

The script uses **1:1 mapping** when running the full test suite:

1. **Discovery**: `pytest <test_file> --collect-only -q` prints one full node id per line (e.g. `path::Class::test_method` or `path::Class::test_method[param]` for parametrized tests).
2. **Parsing**: The script collects every line that looks like a node id and skips header lines (e.g. "collected N items", "Running N items").
3. **Execution**: For each node id, the script runs `pytest <node_id>` from the PyTorch path. So each collected item is run exactly once—no collapsing of parametrized variants.

That way the number of tests discovered matches the number of test runs.
