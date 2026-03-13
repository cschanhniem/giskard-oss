# Top-level conftest for giskard-checks (pytest rootdir).
# Register the eval hook plugin (pytest_generate_tests for @eval_file parametrization).
pytest_plugins = ["giskard.checks.utils.eval"]
