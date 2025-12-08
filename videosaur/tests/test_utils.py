import pytest

from videosaur.videosaur import utils


@pytest.mark.parametrize(
    "inp,path,value,expected",
    [
        ({"a": 1}, "a", 2, {"a": 2}),
        ({"a": {"b": 1}}, "a.b", 2, {"a": {"b": 2}}),
        ({"a": [1, 2, 3]}, "a.0", 4, {"a": [4, 2, 3]}),
    ],
)
def test_write_path(inp, path, value, expected):
    utils.write_path(inp, path, value)
    assert inp == expected
