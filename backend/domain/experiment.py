"""Experiment and generator configuration models.

These ensure reproducibility by recording all parameters needed
to recreate an ML experiment or synthetic data generation run.

See CODING_CONVENTIONS.md — ML section and CONTEXT.md — Reproducibility.
"""


from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for a reproducible ML experiment.

    Temporal splits are mandatory: ``train_time_steps`` must contain
    earlier time steps and ``test_time_steps`` must contain later ones.
    Random shuffling across time steps is a data-leakage defect.
    """

    experiment_id: str = Field(description="Unique identifier for this experiment run")
    dataset_version: str = Field(description="Version/identifier of the dataset used")
    feature_configuration: str = Field(
        description="Feature set identifier (e.g. 'M1_blockchain_only', 'M2_blockchain_graph')"
    )
    split_strategy: str = Field(
        default="temporal",
        description="Must be 'temporal' for primary experiments",
    )
    random_seed: int = Field(description="Random seed for reproducibility")
    train_time_steps: list[int] = Field(
        description="Time steps used for training (must be earlier steps)"
    )
    test_time_steps: list[int] = Field(
        description="Time steps used for testing (must be later steps)"
    )
    model_version: str | None = Field(
        default=None, description="Model version identifier, set after training"
    )


class GeneratorConfig(BaseModel):
    """Configuration for a reproducible synthetic network data generation run.

    Records all parameters needed to reproduce the exact same
    synthetic network observations.
    """

    random_seed: int = Field(description="Random seed for reproducibility")
    generator_version: str = Field(description="Version of the generator module")
    scenario_configs: dict = Field(
        default_factory=dict,
        description="Per-scenario configuration parameters",
    )
    node_pool_size: int | None = Field(
        default=None, description="Number of synthetic network nodes to generate"
    )
