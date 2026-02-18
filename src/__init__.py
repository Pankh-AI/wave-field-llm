"""
Wave Field LLM - Physics-Based Language Model with Wave Field Dynamics
"""

__version__ = "3.5.0"

from .causal_field_attention import CausalFieldAttentionV2
from .causal_field_transformer import CausalFieldTransformer
from .global_context import GlobalContextModule
from .wave_field_attention import WaveFieldAttention
from .wave_field_transformer import WaveFieldTransformer, FieldInterferenceModule

__all__ = [
    "CausalFieldAttentionV2",
    "CausalFieldTransformer",
    "GlobalContextModule",
    "WaveFieldAttention",
    "WaveFieldTransformer",
    "FieldInterferenceModule",
]
