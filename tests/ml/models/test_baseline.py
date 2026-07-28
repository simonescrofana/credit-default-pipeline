"""Test the baseline model factory module.

Provide unit tests to verify instance independence, default parameter
assignments, and custom hyperparameter overrides for the baseline model builder.

"""

from ml.models.baseline import build_baseline_model


def test_build_baseline_model_returns_independent_instances() -> None:
    """Verify that consecutive calls to the factory return distinct instances."""
    model_1 = build_baseline_model()
    model_2 = build_baseline_model()

    assert model_1 is not model_2


def test_build_baseline_model_applies_default_params() -> None:
    """Verify that the model factory applies the expected default parameters."""
    model = build_baseline_model()

    assert model.C == 1
    assert model.class_weight == "balanced"
    assert model.solver == "lbfgs"
    assert model.max_iter == 1000


def test_build_baseline_model_accepts_custom_params() -> None:
    """Verify that custom hyperparameter overrides are correctly applied."""
    model = build_baseline_model(C=0.1, class_weight={0: 1.0, 1: 10.0}, max_iter=2000)

    assert model.C == 0.1
    assert model.class_weight == {0: 1.0, 1: 10.0}
    assert model.max_iter == 2000
