"""
ETL pipeline for enterprise sales data sources.

Ingests data from CSV files, SQL databases, PDFs, emails (EML/MSG),
and call transcripts into a unified list of ``Document`` objects
suitable for ingestion into a vector database / RAG pipeline.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Normalised document record produced by an ETL extractor.

    Attributes:
        doc_id:   Unique identifier for deduplication.
        source:   Origin of the document (file path, URL, DB table, etc.).
        content:  Plain-text content to be embedded.
        metadata: Arbitrary key-value pairs preserved alongside the content.
    """

    doc_id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base for all ETL extractors."""

    source_type: str = "base"

    @abstractmethod
    def extract(self) -> List[Document]:
        """Extract and return a list of normalised ``Document`` objects."""


# ---------------------------------------------------------------------------
# CSV extractor
# ---------------------------------------------------------------------------


class CSVExtractor(BaseExtractor):
    """
    Extracts rows from one or more CSV files as documents.

    Each row is converted to a key=value formatted string so that the
    LLM can reason over structured deal/contact/pipeline data.

    Args:
        paths: Iterable of file paths to process.
        id_column: Column name to use as document ID (falls back to row index).
        encoding: File encoding (default ``"utf-8"``).
    """

    source_type = "csv"

    def __init__(
        self,
        paths: Iterable[Union[str, Path]],
        id_column: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.id_column = id_column
        self.encoding = encoding

    def extract(self) -> List[Document]:
        docs: List[Document] = []
        for path in self.paths:
            with path.open(encoding=self.encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    doc_id = (
                        str(row.get(self.id_column, ""))
                        if self.id_column and self.id_column in row
                        else f"{path.name}:row{idx}"
                    )
                    content = "\n".join(f"{k}: {v}" for k, v in row.items())
                    docs.append(
                        Document(
                            doc_id=doc_id,
                            source=str(path),
                            content=content,
                            metadata={"file": path.name, "row": idx, **dict(row)},
                        )
                    )
        return docs


# ---------------------------------------------------------------------------
# SQL extractor
# ---------------------------------------------------------------------------


class SQLExtractor(BaseExtractor):
    """
    Extracts rows from SQL tables via SQLAlchemy.

    Args:
        connection_url: SQLAlchemy connection URL,
            e.g. ``"postgresql://user:pass@host/db"``
        queries: Mapping of logical name → SQL SELECT statement.
            Each query result set becomes a series of documents.
    """

    source_type = "sql"

    def __init__(
        self,
        connection_url: str,
        queries: Dict[str, str],
    ) -> None:
        self.connection_url = connection_url
        self.queries = queries

    def extract(self) -> List[Document]:
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except ImportError:
            raise RuntimeError(
                "SQLAlchemy is required for SQL extraction. "
                "Install with: pip install sqlalchemy"
            )
        engine = create_engine(self.connection_url)
        docs: List[Document] = []
        with engine.connect() as conn:
            for name, query in self.queries.items():
                result = conn.execute(text(query))
                columns = list(result.keys())
                for idx, row in enumerate(result):
                    row_dict = dict(zip(columns, row))
                    content = "\n".join(f"{k}: {v}" for k, v in row_dict.items())
                    docs.append(
                        Document(
                            doc_id=f"{name}:row{idx}",
                            source=f"sql:{name}",
                            content=content,
                            metadata={"query_name": name, "row": idx, **{k: str(v) for k, v in row_dict.items()}},
                        )
                    )
        return docs


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------


class PDFExtractor(BaseExtractor):
    """
    Extracts text from PDF files (contracts, proposals, RFPs, etc.).

    Uses ``pypdf`` (pure-Python) with an optional ``pdfminer.six`` fallback
    for complex layouts.

    Args:
        paths: Iterable of PDF file paths.
        chunk_size: Approximate character size of each text chunk document.
    """

    source_type = "pdf"

    def __init__(
        self,
        paths: Iterable[Union[str, Path]],
        chunk_size: int = 2000,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.chunk_size = chunk_size

    def _extract_with_pypdf(self, path: Path) -> str:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    def _chunk(self, text: str, source: str) -> List[Document]:
        docs: List[Document] = []
        for i in range(0, max(len(text), 1), self.chunk_size):
            chunk = text[i: i + self.chunk_size]
            if chunk.strip():
                docs.append(
                    Document(
                        doc_id=f"{source}:chunk{i // self.chunk_size}",
                        source=source,
                        content=chunk,
                        metadata={"chunk_index": i // self.chunk_size},
                    )
                )
        return docs

    def extract(self) -> List[Document]:
        docs: List[Document] = []
        for path in self.paths:
            try:
                text = self._extract_with_pypdf(path)
            except ImportError:
                raise RuntimeError(
                    "pypdf is required for PDF extraction. "
                    "Install with: pip install pypdf"
                )
            except Exception as exc:
                logger.warning("PDF extraction failed for %s: %s", path, exc)
                continue
            docs.extend(self._chunk(text, str(path)))
        return docs


# ---------------------------------------------------------------------------
# Email extractor (.eml / .msg)
# ---------------------------------------------------------------------------


class EmailExtractor(BaseExtractor):
    """
    Extracts text content from email files (.eml) or raw email strings.

    Args:
        sources: Iterable of file paths (.eml) or raw RFC-822 strings.
    """

    source_type = "email"

    def __init__(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> None:
        self.sources = list(sources)

    def _parse_email(self, source: Union[str, Path]) -> Document:
        import email as _email
        import hashlib

        raw: str
        source_label: str
        if isinstance(source, Path) or (isinstance(source, str) and os.path.isfile(source)):
            path = Path(source)
            raw = path.read_text(encoding="utf-8", errors="replace")
            source_label = str(path)
        else:
            raw = str(source)
            source_label = "raw_email"

        msg = _email.message_from_string(raw)
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        date = msg.get("Date", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")

        content = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}"
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:32]
        return Document(
            doc_id=doc_id,
            source=source_label,
            content=content,
            metadata={"from": sender, "subject": subject, "date": date},
        )

    def extract(self) -> List[Document]:
        docs: List[Document] = []
        for source in self.sources:
            try:
                docs.append(self._parse_email(source))
            except Exception as exc:
                logger.warning("Email extraction error for %s: %s", source, exc)
        return docs


# ---------------------------------------------------------------------------
# Call transcript extractor
# ---------------------------------------------------------------------------


class CallTranscriptExtractor(BaseExtractor):
    """
    Extracts and normalises call transcripts.

    Accepts plain-text transcript files, VTT (WebVTT), SRT subtitle files,
    or raw transcript strings.

    Args:
        sources: Iterable of file paths or raw transcript strings.
        chunk_size: Approximate character size of each chunk document.
    """

    source_type = "call_transcript"

    def __init__(
        self,
        sources: Iterable[Union[str, Path]],
        chunk_size: int = 1500,
    ) -> None:
        self.sources = list(sources)
        self.chunk_size = chunk_size

    @staticmethod
    def _strip_vtt_srt(text: str) -> str:
        """Remove WebVTT/SRT timestamps and formatting markers."""
        import re

        # Remove WEBVTT header
        text = re.sub(r"^WEBVTT.*\n", "", text)
        # Remove SRT/VTT timestamp lines: 00:00:00,000 --> 00:00:01,000
        text = re.sub(
            r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{1,2}:\d{2}:\d{2}[,.]\d{3}",
            "",
            text,
        )
        # Remove sequence numbers
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Collapse blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _read_source(self, source: Union[str, Path]) -> tuple[str, str]:
        """Return (content, source_label)."""
        if isinstance(source, Path) or (
            isinstance(source, str) and os.path.isfile(source)
        ):
            path = Path(source)
            raw = path.read_text(encoding="utf-8", errors="replace")
            suffix = path.suffix.lower()
            if suffix in (".vtt", ".srt"):
                raw = self._strip_vtt_srt(raw)
            return raw, str(path)
        return str(source), "raw_transcript"

    def _chunk(self, text: str, source: str) -> List[Document]:
        docs: List[Document] = []
        for i in range(0, max(len(text), 1), self.chunk_size):
            chunk = text[i: i + self.chunk_size]
            if chunk.strip():
                docs.append(
                    Document(
                        doc_id=f"{source}:chunk{i // self.chunk_size}",
                        source=source,
                        content=chunk,
                        metadata={"chunk_index": i // self.chunk_size},
                    )
                )
        return docs

    def extract(self) -> List[Document]:
        docs: List[Document] = []
        for source in self.sources:
            try:
                content, source_label = self._read_source(source)
                docs.extend(self._chunk(content, source_label))
            except Exception as exc:
                logger.warning(
                    "Call transcript extraction error for %s: %s", source, exc
                )
        return docs


# ---------------------------------------------------------------------------
# ETL Pipeline orchestrator
# ---------------------------------------------------------------------------


class ETLPipeline:
    """
    Orchestrates multiple extractors and pushes documents into a vector store.

    Usage::

        pipeline = ETLPipeline()
        pipeline.add_extractor(CSVExtractor(["deals.csv"]))
        pipeline.add_extractor(PDFExtractor(["contract.pdf"]))
        docs = pipeline.run()

    Args:
        extractors: Optional initial list of ``BaseExtractor`` instances.
    """

    def __init__(self, extractors: Optional[List[BaseExtractor]] = None) -> None:
        self._extractors: List[BaseExtractor] = list(extractors or [])

    def add_extractor(self, extractor: BaseExtractor) -> "ETLPipeline":
        """Register an extractor.  Returns *self* for chaining."""
        self._extractors.append(extractor)
        return self

    def run(self) -> List[Document]:
        """Execute all extractors and return the aggregated document list."""
        all_docs: List[Document] = []
        for extractor in self._extractors:
            logger.info("Running ETL extractor: %s", extractor.source_type)
            try:
                docs = extractor.extract()
                logger.info(
                    "  → extracted %d documents from %s",
                    len(docs),
                    extractor.source_type,
                )
                all_docs.extend(docs)
            except Exception as exc:
                logger.error("ETL extractor %s failed: %s", extractor.source_type, exc)
        # Deduplicate by doc_id (last write wins)
        seen: Dict[str, Document] = {}
        for doc in all_docs:
            seen[doc.doc_id] = doc
        deduped = list(seen.values())
        logger.info(
            "ETL pipeline complete: %d unique documents extracted.", len(deduped)
        )
        return deduped

    def ingest_to_vectordb(
        self,
        docs: List[Document],
        vectordb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ingest extracted documents into the configured vector database.

        Uses ``langchain_community.vectorstores.Chroma`` by default.

        Args:
            docs: Documents returned by :meth:`run`.
            vectordb_config: Optional override for vector DB settings.
        """
        try:
            from langchain_community.vectorstores import Chroma  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
            from langchain_core.documents import Document as LCDocument  # type: ignore
        except ImportError:
            raise RuntimeError(
                "langchain-community is required for vector DB ingestion. "
                "Install with: pip install langchain-community"
            )

        cfg = vectordb_config or {}
        embed_model = cfg.get("embedding_model", "BAAI/bge-large-en-v1.5")
        persist_dir = cfg.get(
            "persist_directory",
            os.path.join(os.path.expanduser("~"), ".sales_llm", "vectordb"),
        )
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        lc_docs = [
            LCDocument(page_content=doc.content, metadata=doc.metadata)
            for doc in docs
        ]
        vectordb = Chroma.from_documents(
            documents=lc_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        vectordb.persist()
        logger.info(
            "Ingested %d documents into Chroma at %s", len(lc_docs), persist_dir
        )


def build_pipeline_from_config(config: Dict[str, Any]) -> ETLPipeline:
    """
    Factory that constructs an ``ETLPipeline`` from a configuration dictionary.

    Config shape example::

        {
            "csv": [{"paths": ["deals.csv"], "id_column": "deal_id"}],
            "sql": [{"connection_url": "...", "queries": {"deals": "SELECT ..."}}],
            "pdf": [{"paths": ["contract.pdf"]}],
            "email": [{"sources": ["inbox/*.eml"]}],
            "call_transcript": [{"sources": ["transcripts/"]}],
        }
    """
    import glob as _glob

    pipeline = ETLPipeline()

    for entry in config.get("csv", []):
        pipeline.add_extractor(
            CSVExtractor(
                paths=entry["paths"],
                id_column=entry.get("id_column"),
                encoding=entry.get("encoding", "utf-8"),
            )
        )

    for entry in config.get("sql", []):
        pipeline.add_extractor(
            SQLExtractor(
                connection_url=entry["connection_url"],
                queries=entry["queries"],
            )
        )

    for entry in config.get("pdf", []):
        expanded = []
        for p in entry["paths"]:
            expanded.extend(_glob.glob(p, recursive=True) or [p])
        pipeline.add_extractor(
            PDFExtractor(paths=expanded, chunk_size=entry.get("chunk_size", 2000))
        )

    for entry in config.get("email", []):
        expanded = []
        for s in entry["sources"]:
            expanded.extend(_glob.glob(s, recursive=True) or [s])
        pipeline.add_extractor(EmailExtractor(sources=expanded))

    for entry in config.get("call_transcript", []):
        expanded = []
        for s in entry["sources"]:
            expanded.extend(_glob.glob(s, recursive=True) or [s])
        pipeline.add_extractor(
            CallTranscriptExtractor(
                sources=expanded,
                chunk_size=entry.get("chunk_size", 1500),
            )
        )

    return pipeline
