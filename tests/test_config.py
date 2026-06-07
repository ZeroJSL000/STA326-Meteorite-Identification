from dataclasses import replace

import pytest

from meteorite_final.config import PROJECT_ROOT, InferenceConfig, project_path


def test_project_path_is_root_relative() -> None:
    assert project_path("data") == (PROJECT_ROOT / "data").resolve()


def test_config_rejects_invalid_amp_dtype() -> None:
    config = InferenceConfig(model_name="model")
    with pytest.raises(ValueError, match="amp_dtype"):
        replace(config, amp_dtype="float32").validate()
