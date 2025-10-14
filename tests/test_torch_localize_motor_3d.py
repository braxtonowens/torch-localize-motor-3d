import torch_localize_motor_3d


def test_imports_with_version():
    assert isinstance(torch_localize_motor_3d.__version__, str)
