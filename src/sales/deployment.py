"""
Deployment helpers for the enterprise sales LLM.

Generates Docker Compose and Kubernetes manifest snippets for
on-prem / VPC deployments with GPU and CPU fallback modes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from h2ogpt.src.sales.config import DeploymentMode, SalesLLMConfig


# ---------------------------------------------------------------------------
# Docker Compose generator
# ---------------------------------------------------------------------------


def generate_docker_compose(
    config: SalesLLMConfig,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a ``docker-compose.yml`` snippet for the sales LLM service.

    The generated compose file includes:
    - The core h2oGPT sales LLM service (GPU or CPU depending on config).
    - A Chroma vector database service.
    - A Redis service for persistent memory (if configured).
    - An nginx reverse proxy for TLS termination.

    Args:
        config: :class:`SalesLLMConfig` instance.
        output_path: If provided, write the YAML to this path.

    Returns:
        YAML string.
    """
    gpu_section = ""
    if config.model.gpu_enabled:
        gpu_section = """\
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]"""

    compose = f"""\
version: "3.9"

services:
  sales-llm:
    image: h2oai/h2ogpt:latest
    restart: always
    shm_size: "4gb"
    ports:
      - "{config.api_port}:{config.api_port}"
    environment:
      - SALES_LLM_BASE_MODEL={config.model.base_model}
      - SALES_LLM_CONTEXT_LENGTH={config.model.context_length.value}
      - SALES_LLM_STREAMING={str(config.model.streaming).lower()}
      - SALES_LLM_GPU_ENABLED={str(config.model.gpu_enabled).lower()}
      - SALES_LLM_DEPLOYMENT_MODE={config.deployment_mode.value}
      - SALES_LLM_API_PORT={config.api_port}
      - SALES_MEMORY_URL=redis://redis:6379/0
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    volumes:
      - sales-llm-cache:/workspace/.cache
      - sales-llm-save:/workspace/save
      - ./data:/workspace/data:ro
    depends_on:
      - chromadb
      - redis
    deploy:
{gpu_section if gpu_section else "      {}"}

  chromadb:
    image: chromadb/chroma:latest
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma

  redis:
    image: redis:7-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --save 60 1 --loglevel warning

  nginx:
    image: nginx:stable-alpine
    restart: always
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - sales-llm

volumes:
  sales-llm-cache:
  sales-llm-save:
  chroma-data:
  redis-data:
"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(compose)

    return compose


# ---------------------------------------------------------------------------
# Kubernetes manifest generator
# ---------------------------------------------------------------------------


def generate_kubernetes_manifest(
    config: SalesLLMConfig,
    namespace: str = "sales-llm",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a Kubernetes manifest for the sales LLM deployment.

    The generated manifest includes:
    - Namespace
    - Deployment (GPU toleration + node selector when GPU is enabled)
    - Service (ClusterIP)
    - HorizontalPodAutoscaler
    - ConfigMap with runtime environment variables

    Args:
        config: :class:`SalesLLMConfig` instance.
        namespace: Kubernetes namespace.
        output_path: If provided, write the YAML to this path.

    Returns:
        YAML string (multiple documents separated by ``---``).
    """
    gpu_resources = ""
    gpu_toleration = ""
    if config.model.gpu_enabled:
        gpu_resources = """\
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1\""""
        gpu_toleration = """\
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      nodeSelector:
        accelerator: nvidia-tesla-a100"""

    manifest = f"""\
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: sales-llm-config
  namespace: {namespace}
data:
  SALES_LLM_BASE_MODEL: "{config.model.base_model}"
  SALES_LLM_CONTEXT_LENGTH: "{config.model.context_length.value}"
  SALES_LLM_STREAMING: "{str(config.model.streaming).lower()}"
  SALES_LLM_GPU_ENABLED: "{str(config.model.gpu_enabled).lower()}"
  SALES_LLM_DEPLOYMENT_MODE: "{config.deployment_mode.value}"
  SALES_LLM_API_PORT: "{config.api_port}"
  SALES_MEMORY_URL: "redis://redis-service.{namespace}.svc.cluster.local:6379/0"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sales-llm
  namespace: {namespace}
  labels:
    app: sales-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sales-llm
  template:
    metadata:
      labels:
        app: sales-llm
    spec:
{gpu_toleration}
      containers:
      - name: sales-llm
        image: h2oai/h2ogpt:latest
        ports:
        - containerPort: {config.api_port}
        envFrom:
        - configMapRef:
            name: sales-llm-config
        volumeMounts:
        - name: model-cache
          mountPath: /workspace/.cache
{gpu_resources}
        livenessProbe:
          httpGet:
            path: /health
            port: {config.api_port}
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: {config.api_port}
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: sales-llm-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: sales-llm-service
  namespace: {namespace}
spec:
  selector:
    app: sales-llm
  ports:
  - protocol: TCP
    port: 80
    targetPort: {config.api_port}
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sales-llm-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sales-llm
  minReplicas: 1
  maxReplicas: 4
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(manifest)

    return manifest


# ---------------------------------------------------------------------------
# Runtime health-check helper
# ---------------------------------------------------------------------------


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability for the sales LLM.

    Returns a dict with ``available`` (bool), ``device_count`` (int),
    and ``device_names`` (list of str).
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return {"available": True, "device_count": count, "device_names": names}
    except ImportError:
        pass
    return {"available": False, "device_count": 0, "device_names": []}


def get_deployment_summary(config: SalesLLMConfig) -> str:
    """Return a human-readable deployment summary string."""
    gpu_info = check_gpu_availability()
    gpu_status = (
        f"{gpu_info['device_count']} × {', '.join(gpu_info['device_names'])}"
        if gpu_info["available"]
        else "CPU only"
    )
    return (
        f"Sales LLM Deployment Summary\n"
        f"{'=' * 40}\n"
        f"Model          : {config.model.base_model}\n"
        f"Model size     : {config.model.model_size.value}\n"
        f"Context length : {config.model.context_length.value:,} tokens\n"
        f"Quantization   : {config.model.quantization}\n"
        f"GPU            : {gpu_status}\n"
        f"Deployment mode: {config.deployment_mode.value}\n"
        f"API port       : {config.api_port}\n"
        f"RAG backend    : {config.rag.vector_db}\n"
        f"Memory backend : {config.memory.backend}\n"
        f"CRM connectors : {', '.join(config.crm_connectors)}\n"
        f"Comms          : {', '.join(config.communication_connectors)}\n"
        f"ETL sources    : {', '.join(config.etl_sources)}\n"
        f"Streaming      : {config.model.streaming}\n"
        f"Target latency : {config.model.target_latency_seconds}s\n"
    )
