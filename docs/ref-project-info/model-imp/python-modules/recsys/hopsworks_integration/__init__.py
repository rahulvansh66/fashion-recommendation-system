"""
⚠️ REFERENCE PROJECT DISCLAIMER ⚠️

THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION

- DO NOT USE unless explicitly asked to reference old code
- CURRENT IMPLEMENTATION is in system-design/ directory
- This file is for REFERENCE ONLY to understand legacy approaches
- All new development should follow current system design specifications
"""

from . import feature_store, ranking_serving, two_tower_serving, llm_ranking_serving
from .feature_store import get_feature_store

__all__ = ["feature_store", "get_feature_store", "ranking_serving", "two_tower_serving", "llm_ranking_serving"]

"""
⚠️ END OF REFERENCE PROJECT FILE ⚠️

Remember: This is archived code. Use system-design/ for current implementation.
"""
