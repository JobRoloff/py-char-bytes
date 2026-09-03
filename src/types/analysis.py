from pydantic import BaseModel, ConfigDict, Field


# -------------------------------------------------------------------
# Sub-Models
# -------------------------------------------------------------------
class ReadabilityMetrics(BaseModel):
    flesch_reading_ease: float = Field(..., description="Flesch Reading Ease score (0 to 100)")
    gunning_fog: float = Field(..., description="Gunning Fog Index grade level")
    coleman_liau_index: float = Field(..., description="Coleman-Liau character-based grade level")
    dale_chall_score: float = Field(..., description="Dale-Chall vocabulary lookup score")


class SyntacticMetrics(BaseModel):
    word_count: int = Field(..., ge=0, description="Total document token count")
    passive_voice_count: int = Field(..., ge=0, description="Count of passive auxiliary tokens")
    noun_to_verb_ratio: float = Field(..., ge=0.0, description="Structural ratio of nouns to verbs")


class GemmaClarityResponse(BaseModel):
    """Schema enforced on the local Ollama/Gemma SLM output."""

    tone: str = Field(..., description="Categorical tone classification (e.g., Academic, Informal)")
    clarity_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Qualitative clarity rating bounded strictly between 1 and 10",
    )
    summary: str = Field(..., description="1-sentence summary of core argument")


class SystemMetrics(BaseModel):
    """Runtime execution & performance metadata."""

    latency_ms: float = Field(
        ..., ge=0.0, description="End-to-end processing latency in milliseconds"
    )
    schema_valid: bool = Field(
        default=False, description="Flag indicating if SLM returned valid JSON matching schema"
    )


# -------------------------------------------------------------------
# Top-Level Pipeline Results
# -------------------------------------------------------------------
class DeterministicAnalysis(BaseModel):
    readability: ReadabilityMetrics
    syntactic: SyntacticMetrics


class PipelineResult(BaseModel):
    document_id: str
    deterministic: DeterministicAnalysis
    slm_insights: GemmaClarityResponse | None = Field(
        default=None,
        description="SLM evaluation output; None if inference failed or schema was invalid",
    )
    system: SystemMetrics

    model_config = ConfigDict(extra="ignore")

    def to_flat_dict(self) -> dict:
        """Helper method to flatten nested fields into a single 1D dict for Pandas/Polars conversion."""
        flat = {
            "document_id": self.document_id,
            "latency_ms": self.system.latency_ms,
            "schema_valid": self.system.schema_valid,
            # Syntactic
            "word_count": self.deterministic.syntactic.word_count,
            "passive_voice_count": self.deterministic.syntactic.passive_voice_count,
            "noun_to_verb_ratio": self.deterministic.syntactic.noun_to_verb_ratio,
            # Readability
            "flesch_reading_ease": self.deterministic.readability.flesch_reading_ease,
            "gunning_fog": self.deterministic.readability.gunning_fog,
            "coleman_liau_index": self.deterministic.readability.coleman_liau_index,
            "dale_chall_score": self.deterministic.readability.dale_chall_score,
            # SLM Insights (Optional)
            "gemma_clarity_score": self.slm_insights.clarity_score if self.slm_insights else None,
            "gemma_tone": self.slm_insights.tone if self.slm_insights else None,
            "gemma_summary": self.slm_insights.summary if self.slm_insights else None,
        }
        return flat
