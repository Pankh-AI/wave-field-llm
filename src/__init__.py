"""
Wave Field LLM - Physics-Based Language Model with Wave Field Dynamics
"""

__version__ = "4.3.6"

from .global_context import GlobalContextModule
from .wave_field_attention import WaveFieldAttention
from .wave_field_transformer import WaveFieldTransformer, FieldInterferenceModule

# Legacy V1/V2 imports (moved to src/legacy/)
try:
    from .legacy.causal_field_attention import CausalFieldAttentionV2
    from .legacy.causal_field_transformer import CausalFieldTransformer
except ImportError:
    CausalFieldAttentionV2 = None
    CausalFieldTransformer = None

__all__ = [
    "GlobalContextModule",
    "WaveFieldAttention",
    "WaveFieldTransformer",
    "FieldInterferenceModule",
]
