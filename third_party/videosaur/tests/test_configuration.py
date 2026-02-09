from third_party.videosaur.videosaur import configuration


def test_resolver_eval():
    # Version with full lambda function specification
    fn = "lambda x, y: x + y"
    assert configuration.resolver_eval(fn, 1, 2) == 3

    # Version without lambda prefix
    fn = "x, y, z: x + y + z"
    assert configuration.resolver_eval(fn, 1, 2, 3) == 6

    # Version with auto arguments
    fn = "a + b + c + d"
    assert configuration.resolver_eval(fn, 1, 2, 3, 4) == 10
