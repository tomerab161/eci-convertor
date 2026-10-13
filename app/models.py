from pydantic import BaseModel, ConfigDict, Field


class OrbitalPoint(BaseModel):
    """A single ECI state vector."""

    model_config = ConfigDict(extra="ignore")

    MjdTime: float = Field(description="Modified Julian Date (UTC)")
    Vx: float = Field(description="Velocity X (ECI)")
    Vy: float = Field(description="Velocity Y (ECI)")
    Vz: float = Field(description="Velocity Z (ECI)")
    x: float = Field(description="Position X (ECI)")
    y: float = Field(description="Position Y (ECI)")
    z: float = Field(description="Position Z (ECI)")
