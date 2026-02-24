"""
Enterprise-grade Sales LLM module for h2oGPT.

Provides CRM integrations, ETL pipelines, sales intelligence,
security controls, and deployment helpers for high-value B2B sales
workflows (₹50L–₹2Cr+ deals).
"""

from h2ogpt.src.sales.config import SalesLLMConfig, ModelSize, DeploymentMode
from h2ogpt.src.sales.intelligence import SalesIntelligence
from h2ogpt.src.sales.security import RBACManager, AuditLogger

__all__ = [
    "SalesLLMConfig",
    "ModelSize",
    "DeploymentMode",
    "SalesIntelligence",
    "RBACManager",
    "AuditLogger",
]
