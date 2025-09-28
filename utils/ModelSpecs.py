from typing import Literal, ClassVar, Dict
from pydantic import BaseModel, Field

VariantName = Literal["toy", "tiny", "small", "target"]

class ModelSpecs(BaseModel):
    VOCAB_SIZE: int
    BLOCK_SIZE: int
    N_EMBD: int
    N_HEAD: int
    N_LAYER: int
    DROPOUT: float = Field(..., ge=0.0, le=1.0)

    variants: ClassVar[Dict[str, dict]] = {
        "toy": dict(VOCAB_SIZE=500, BLOCK_SIZE=64, N_EMBD=128, N_HEAD=4, N_LAYER=3, DROPOUT=0.1),
        "tiny": dict(VOCAB_SIZE=1000, BLOCK_SIZE=128, N_EMBD=256, N_HEAD=6, N_LAYER=6, DROPOUT=0.15),
        "small": dict(VOCAB_SIZE=1800, BLOCK_SIZE=128, N_EMBD=384, N_HEAD=6, N_LAYER=6, DROPOUT=0.2),
        "target": dict(VOCAB_SIZE=3200, BLOCK_SIZE=256, N_EMBD=512, N_HEAD=8, N_LAYER=8, DROPOUT=0.25),
    }

    @classmethod
    def create(cls, variant_name: VariantName, **overrides) -> "ModelSpecs":
        variant = cls.variants.get(variant_name)
        if variant is None:
            raise ValueError(f"Unknown variant '{variant_name}'. Valid options: {list(cls.variants.keys())}")
        data = {**variant, **overrides}
        return cls.model_validate(data)
