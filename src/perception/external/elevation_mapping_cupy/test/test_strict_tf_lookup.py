"""Exercise the actual node TF method without importing ROS/CuPy or a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

class Extrapolation(Exception):
    pass

class RosExtrapolation(Exception):
    pass

class Lookup(Exception):
    pass

class Connectivity(Exception):
    pass

def method_under_test():
    source = Path(__file__).resolve().parents[1] / "elevation_mapping_cupy" / "elevation_mapping_node.py"
    tree = ast.parse(source.read_text())
    node_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ElevationMappingNode")
    method = next(n for n in node_class.body if isinstance(n, ast.FunctionDef) and n.name == "safe_lookup_transform")
    # Compile the production method itself; no copied implementation or GPU imports.
    scope = {
        "tf2": SimpleNamespace(ExtrapolationException=Extrapolation, LookupException=Lookup, ConnectivityException=Connectivity),
        "tf2_ros": SimpleNamespace(ExtrapolationException=RosExtrapolation),
        "rclpy": SimpleNamespace(time=SimpleNamespace(Time=lambda: "latest")),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
    return scope["safe_lookup_transform"]

class Buffer:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.transform = object()
    def lookup_transform(self, target, source, stamp):
        self.calls.append((target, source, stamp))
        if len(self.calls) == 1 and self.failure:
            raise self.failure("unavailable at requested time")
        return self.transform

def node_with(buffer):
    warnings = []
    return SimpleNamespace(
        _tf_buffer=buffer,
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: warnings.append((args, kwargs))),
    ), warnings

def test_success_preserves_exact_acquisition_stamp():
    stamp = object()
    buffer = Buffer()
    node, warnings = node_with(buffer)
    assert method_under_test()(node, "map", "rslidar", stamp) is buffer.transform
    assert buffer.calls == [("map", "rslidar", stamp)]
    assert warnings == []

@pytest.mark.parametrize("error", [Extrapolation, RosExtrapolation, Lookup, Connectivity])
def test_missing_timestamp_never_retries_latest(error):
    stamp = object()
    buffer = Buffer(error)
    node, warnings = node_with(buffer)
    assert method_under_test()(node, "map", "rslidar", stamp) is None
    assert buffer.calls == [("map", "rslidar", stamp)]
    assert len(warnings) == 1

def test_explicit_latest_time_is_still_allowed():
    zero_time = object()  # The caller owns the intentional Time(0) request.
    buffer = Buffer()
    node, warnings = node_with(buffer)
    assert method_under_test()(node, "map", "base_link", zero_time) is buffer.transform
    assert buffer.calls == [("map", "base_link", zero_time)]
    assert warnings == []
