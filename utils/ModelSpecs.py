from typing import Literal, ClassVar, Dict, Optional
from pydantic import BaseModel, Field

VariantName = Literal["toy", "tiny", "small", "mid", "target", 'dream']

class ModelSpecs(BaseModel):
    VOCAB_SIZE: int # total number of tokens
    BLOCK_SIZE: int # what is the maximum context length for predictions?
    N_EMBD: int
    N_HEAD: int
    N_LAYER: int
    DROPOUT: float = Field(..., ge=0.0, le=1.0)
    KV_HEADS: int

    variants: ClassVar[Dict[str, dict]] = {
        "toy": dict(VOCAB_SIZE=50261, BLOCK_SIZE=64, N_EMBD=128, N_HEAD=4, N_LAYER=3, DROPOUT=0.1, KV_HEADS=8),
        "tiny": dict(VOCAB_SIZE=50261, BLOCK_SIZE=128, N_EMBD=256, N_HEAD=6, N_LAYER=6, DROPOUT=0.15, KV_HEADS=8),
        "small": dict(VOCAB_SIZE=50261, BLOCK_SIZE=256, N_EMBD=384, N_HEAD=6, N_LAYER=6, DROPOUT=0.2, KV_HEADS=8), # andrej kerpathy model
        # "mid": dict(VOCAB_SIZE=50261, BLOCK_SIZE=512, N_EMBD=512, N_HEAD=8, N_LAYER=8, DROPOUT=0.25, KV_HEADS=8),
        "mid": dict(VOCAB_SIZE=50261, BLOCK_SIZE=512, N_EMBD=512, N_HEAD=8, N_LAYER=8, DROPOUT=0.25, KV_HEADS=8),
        # "target": dict(VOCAB_SIZE=50261, BLOCK_SIZE=512, N_EMBD=512, N_HEAD=12, N_LAYER=16, DROPOUT=0.30, KV_HEADS=8),
        "target": dict(VOCAB_SIZE=50261, BLOCK_SIZE=512, N_EMBD=516, N_HEAD=12, N_LAYER=16, DROPOUT=0.30, KV_HEADS=8),
        "dream": dict(VOCAB_SIZE=50261, BLOCK_SIZE=1024, N_EMBD=516, N_HEAD=12, N_LAYER=16, DROPOUT=0.30, KV_HEADS=8),
    }
    # NOTE for our target, we can experiment with the embedding size to be till 768 also
    # KV_HEADS is a parameter specific to W-GQA, and 4 and 8 heads represent various performance tradeoffs, prefer 8 for now.

    @classmethod
    def create(cls, variant_name: VariantName, **overrides) -> "ModelSpecs":
        variant = cls.variants.get(variant_name)
        if variant is None:
            raise ValueError(f"Unknown variant '{variant_name}'. Valid options: {list(cls.variants.keys())}")
        data = {**variant, **overrides}
        return cls.model_validate(data)
