"""
Enterprise Sales LLM configuration.

Supports 30B–70B parameter models with 64K–128K token context, GPU/CPU
fallback, streaming inference, and sub-1.5 s latency targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelSize(str, Enum):
    """Supported enterprise model sizes."""

    B_30 = "30b"
    B_70 = "70b"


class ContextLength(int, Enum):
    """Supported context window sizes (tokens)."""

    K_64 = 65536
    K_128 = 131072


class DeploymentMode(str, Enum):
    """Target deployment environment."""

    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    ON_PREM = "on_prem"
    VPC = "vpc"


@dataclass
class ModelConfig:
    """Core model configuration for enterprise sales workloads."""

    model_size: ModelSize = ModelSize.B_70
    context_length: ContextLength = ContextLength.K_128
    # Quantization: "none" | "int8" | "int4"
    quantization: str = "int8"
    # Use GPU when available, fall back to CPU automatically
    gpu_enabled: bool = True
    cpu_fallback: bool = True
    # Number of GPUs to use (0 = auto-detect)
    num_gpus: int = 0
    # Streaming inference
    streaming: bool = True
    # Target latency in seconds
    target_latency_seconds: float = 1.5
    # Maximum concurrent requests
    max_concurrent_requests: int = 32
    # Base model name / HuggingFace path
    base_model: str = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class RAGConfig:
    """Retrieval-Augmented Generation settings."""

    enabled: bool = True
    # Vector database backend: "chroma" | "weaviate" | "faiss" | "qdrant"
    vector_db: str = "chroma"
    # Embedding model
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # Number of documents retrieved per query
    top_k: int = 10
    # Enable cross-encoder reranking for precision
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Persist directory for the vector store
    persist_directory: str = os.path.join(os.path.expanduser("~"), ".sales_llm", "vectordb")


@dataclass
class MemoryConfig:
    """Persistent conversation and deal memory."""

    enabled: bool = True
    backend: str = "redis"  # "redis" | "postgres" | "sqlite"
    # Connection URL (supports env-var substitution)
    connection_url: str = os.getenv("SALES_MEMORY_URL", "redis://localhost:6379/0")
    # Maximum turns stored per deal/conversation
    max_turns: int = 1000
    # Time-to-live in seconds (0 = no expiry)
    ttl_seconds: int = 0


@dataclass
class SalesLLMConfig:
    """Top-level configuration for the enterprise sales LLM."""

    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    deployment_mode: DeploymentMode = DeploymentMode.DOCKER
    # List of enabled CRM connectors
    crm_connectors: List[str] = field(
        default_factory=lambda: ["salesforce", "hubspot", "zoho"]
    )
    # List of enabled communication connectors
    communication_connectors: List[str] = field(
        default_factory=lambda: ["gmail", "outlook", "slack", "whatsapp", "zoom", "teams"]
    )
    # List of enabled ETL sources
    etl_sources: List[str] = field(
        default_factory=lambda: ["csv", "sql", "pdf", "email", "call_transcript"]
    )
    # Enable sales intelligence features
    intelligence_enabled: bool = True
    # Enable RBAC and audit logging
    security_enabled: bool = True
    # API host/port for the sales LLM service
    api_host: str = "0.0.0.0"
    api_port: int = 8090
    # Additional keyword overrides passed through to h2oGPT's gen.py
    h2ogpt_kwargs: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "SalesLLMConfig":
        """Build a config from environment variables with safe defaults."""
        cfg = cls()
        if os.getenv("SALES_LLM_MODEL_SIZE"):
            cfg.model.model_size = ModelSize(os.environ["SALES_LLM_MODEL_SIZE"])
        if os.getenv("SALES_LLM_CONTEXT_LENGTH"):
            cfg.model.context_length = ContextLength(
                int(os.environ["SALES_LLM_CONTEXT_LENGTH"])
            )
        if os.getenv("SALES_LLM_BASE_MODEL"):
            cfg.model.base_model = os.environ["SALES_LLM_BASE_MODEL"]
        if os.getenv("SALES_LLM_GPU_ENABLED"):
            cfg.model.gpu_enabled = os.environ["SALES_LLM_GPU_ENABLED"].lower() in (
                "1",
                "true",
                "yes",
            )
        if os.getenv("SALES_LLM_STREAMING"):
            cfg.model.streaming = os.environ["SALES_LLM_STREAMING"].lower() in (
                "1",
                "true",
                "yes",
            )
        if os.getenv("SALES_LLM_DEPLOYMENT_MODE"):
            cfg.deployment_mode = DeploymentMode(os.environ["SALES_LLM_DEPLOYMENT_MODE"])
        if os.getenv("SALES_LLM_API_PORT"):
            cfg.api_port = int(os.environ["SALES_LLM_API_PORT"])
        return cfg
