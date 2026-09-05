"""Independent small-shape oracles for the Qwen4 preview architecture."""

from .common import actual_gguf_grouped_rmsnorm, source_grouped_rmsnorm
from .gated_residual import actual_gguf_final_read as gated_residual_actual_gguf_final_read
from .gated_residual import actual_gguf_read as gated_residual_actual_gguf_read
from .gated_residual import inject as gated_residual_inject
from .gated_residual import source_final_read as gated_residual_source_final_read
from .gated_residual import source_read as gated_residual_source_read
from .ggml_codecs import decode_iq1_s, decode_iq2_xxs, decode_iq4_nl
from .ggml_k_codecs import decode_q4_k, decode_q5_k, decode_q6_k, decode_q8_0
from .gdn import ActualGgufGDNWeights, GDNState, SourceGDNWeights
from .gdn import actual_gguf_sublayer as gdn_actual_gguf_sublayer
from .gdn import output_projection as gdn_output_projection
from .gdn import recurrence as gdn_recurrence
from .gdn import source_sublayer as gdn_source_sublayer
from .moe import sparse_moe
from .ngram import ids as ngram_ids
from .ple import actual_gguf_inject as ple_actual_gguf_inject
from .ple import source_inject as ple_source_inject
from .qsa import actual_gguf_select as qsa_actual_gguf_select
from .qsa import actual_gguf_sparse_attention as qsa_actual_gguf_sparse_attention
from .qsa import source_select as qsa_source_select
from .qsa import source_sparse_attention as qsa_source_sparse_attention
from .vision import patch_merger as vision_patch_merger

__all__ = [
    "actual_gguf_grouped_rmsnorm",
    "gated_residual_actual_gguf_final_read",
    "gated_residual_actual_gguf_read",
    "gated_residual_inject",
    "gated_residual_source_final_read",
    "gated_residual_source_read",
    "decode_iq1_s",
    "decode_iq2_xxs",
    "decode_iq4_nl",
    "decode_q4_k",
    "decode_q5_k",
    "decode_q6_k",
    "decode_q8_0",
    "gdn_output_projection",
    "gdn_recurrence",
    "gdn_actual_gguf_sublayer",
    "gdn_source_sublayer",
    "ActualGgufGDNWeights",
    "GDNState",
    "SourceGDNWeights",
    "ngram_ids",
    "ple_actual_gguf_inject",
    "ple_source_inject",
    "qsa_actual_gguf_select",
    "qsa_actual_gguf_sparse_attention",
    "qsa_source_select",
    "qsa_source_sparse_attention",
    "sparse_moe",
    "source_grouped_rmsnorm",
    "vision_patch_merger",
]
