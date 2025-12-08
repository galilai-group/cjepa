import pytest


@pytest.fixture
def data_root_dir() -> str:
    from videosaur import data

    return data.get_data_root_dir(error_on_missing_path=True)
