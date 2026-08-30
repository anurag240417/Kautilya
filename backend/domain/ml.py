"""Machine learning score domain model.

Predictions should include, when available:
    - prediction
    - confidence/probability
    - model/version identifier
    - relevant features

See ARCHITECTURE.md §ml/ and AGENTS.md §7.
"""


from pydantic import BaseModel, Field


class MLScore(BaseModel):
    """ML prediction record for a transaction or wallet.

    Carries model provenance so downstream consumers can trace
    predictions back to the model version and feature set that
    produced them.
    """

    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(description="'transaction' or 'wallet'")
    prediction: int = Field(description="Predicted class label")
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence/probability for the predicted class",
    )
    anomaly_score: float | None = Field(
        default=None, description="Anomaly detection score, if applicable"
    )
    model_version: str = Field(description="Identifier of the model that produced this score")
    feature_set: str | None = Field(
        default=None,
        description="Feature configuration identifier (e.g. 'M1_blockchain_only')",
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any input to this prediction included synthetic data",
    )
