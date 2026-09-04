"""Independent small-shape oracles for the Qwen4 preview architecture."""

from .common import grouped_zero_centered_rmsnorm
from .gated_residual import final_read as gated_residual_final_read
from .gated_residual import inject as gated_residual_inject
from .gated_residual import read as gated_residual_read
from .ggml_codecs import decode_iq1_s, decode_iq2_xxs, decode_iq4_nl
from .ggml_k_codecs import decode_q4_k, decode_q5_k, decode_q6_k, decode_q8_0
from .gdn import GDNState, GDNWeights
from .gdn import output_projection as gdn_output_projection
from .gdn import recurrence as gdn_recurrence
from .gdn import sublayer as gdn_sublayer
from .moe import sparse_moe
from .ngram import ids as ngram_ids
from .ple import inject as ple_inject
from .qsa import select as qsa_select
from .qsa import sparse_attention as qsa_sparse_attention
from .vision import patch_merger as vision_patch_merger

__all__ = [
    "gated_residual_final_read",
    "gated_residual_inject",
    "gated_residual_read",
    "decode_iq1_s",
    "decode_iq2_xxs",
    "decode_iq4_nl",
    "decode_q4_k",
    "decode_q5_k",
    "decode_q6_k",
    "decode_q8_0",
    "gdn_output_projection",
    "gdn_recurrence",
    "gdn_sublayer",
    "GDNState",
    "GDNWeights",
    "grouped_zero_centered_rmsnorm",
    "ngram_ids",
    "ple_inject",
    "qsa_select",
    "qsa_sparse_attention",
    "sparse_moe",
    "vision_patch_merger",
]
