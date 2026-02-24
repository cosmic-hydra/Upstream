"""
Enterprise security controls for the sales LLM.

Provides:
- Role-Based Access Control (RBAC)
- AES-256 field-level encryption / decryption
- Structured audit logging
- Tenant data isolation helpers
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------


class Permission(str, Enum):
    """Granular permissions for the sales LLM platform."""

    # Data access
    READ_DEALS = "read:deals"
    WRITE_DEALS = "write:deals"
    READ_CONTACTS = "read:contacts"
    WRITE_CONTACTS = "write:contacts"
    READ_ACTIVITIES = "read:activities"
    WRITE_ACTIVITIES = "write:activities"
    # Intelligence
    USE_INTELLIGENCE = "use:intelligence"
    READ_FORECASTS = "read:forecasts"
    # Administration
    MANAGE_USERS = "admin:users"
    MANAGE_ROLES = "admin:roles"
    VIEW_AUDIT_LOG = "admin:audit_log"
    # LLM inference
    RUN_INFERENCE = "llm:inference"
    RUN_AGENTIC = "llm:agentic"


@dataclass
class Role:
    """Named role with a set of permissions."""

    name: str
    permissions: FrozenSet[Permission]
    description: str = ""

    def has_permission(self, perm: Permission) -> bool:
        return perm in self.permissions


# Predefined roles
ROLES: Dict[str, Role] = {
    "admin": Role(
        name="admin",
        permissions=frozenset(Permission),
        description="Full administrative access.",
    ),
    "sales_manager": Role(
        name="sales_manager",
        permissions=frozenset(
            {
                Permission.READ_DEALS,
                Permission.WRITE_DEALS,
                Permission.READ_CONTACTS,
                Permission.WRITE_CONTACTS,
                Permission.READ_ACTIVITIES,
                Permission.WRITE_ACTIVITIES,
                Permission.USE_INTELLIGENCE,
                Permission.READ_FORECASTS,
                Permission.RUN_INFERENCE,
                Permission.RUN_AGENTIC,
                Permission.VIEW_AUDIT_LOG,
            }
        ),
        description="Sales manager with full deal and intelligence access.",
    ),
    "sales_rep": Role(
        name="sales_rep",
        permissions=frozenset(
            {
                Permission.READ_DEALS,
                Permission.WRITE_DEALS,
                Permission.READ_CONTACTS,
                Permission.WRITE_CONTACTS,
                Permission.READ_ACTIVITIES,
                Permission.WRITE_ACTIVITIES,
                Permission.USE_INTELLIGENCE,
                Permission.RUN_INFERENCE,
            }
        ),
        description="Individual contributor with deal and LLM access.",
    ),
    "read_only": Role(
        name="read_only",
        permissions=frozenset(
            {
                Permission.READ_DEALS,
                Permission.READ_CONTACTS,
                Permission.READ_ACTIVITIES,
                Permission.READ_FORECASTS,
            }
        ),
        description="Read-only access for analysts / observers.",
    ),
}


@dataclass
class User:
    """Platform user record."""

    user_id: str
    email: str
    tenant_id: str
    role_name: str
    is_active: bool = True
    _password_hash: str = field(default="", repr=False)

    @property
    def role(self) -> Role:
        """Resolve the user's :class:`Role` object."""
        if self.role_name not in ROLES:
            raise ValueError(f"Unknown role: {self.role_name}")
        return ROLES[self.role_name]

    def has_permission(self, perm: Permission) -> bool:
        """Return True if the user has the given permission."""
        return self.is_active and self.role.has_permission(perm)


class RBACManager:
    """
    Thread-safe RBAC manager.

    Manages user creation, role assignment, and permission checks.
    Uses an in-memory store by default; integrate with a persistent
    backend (e.g. PostgreSQL) by overriding :meth:`_load_user` and
    :meth:`_save_user`.

    Args:
        custom_roles: Optional additional :class:`Role` instances to register.
    """

    def __init__(self, custom_roles: Optional[List[Role]] = None) -> None:
        self._users: Dict[str, User] = {}
        self._lock = threading.Lock()
        self._roles: Dict[str, Role] = dict(ROLES)
        for role in custom_roles or []:
            self._roles[role.name] = role

    def register_role(self, role: Role) -> None:
        """Register a custom role."""
        with self._lock:
            self._roles[role.name] = role

    def create_user(
        self,
        user_id: str,
        email: str,
        tenant_id: str,
        role_name: str,
        password: Optional[str] = None,
    ) -> User:
        """
        Create and register a new user.

        Raises:
            ValueError: If the role does not exist or the user already exists.
        """
        with self._lock:
            if role_name not in self._roles:
                raise ValueError(f"Role '{role_name}' is not registered.")
            if user_id in self._users:
                raise ValueError(f"User '{user_id}' already exists.")
            pw_hash = self._hash_password(password) if password else ""
            user = User(
                user_id=user_id,
                email=email,
                tenant_id=tenant_id,
                role_name=role_name,
                _password_hash=pw_hash,
            )
            self._users[user_id] = user
            return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Return the user for *user_id*, or None if not found."""
        return self._users.get(user_id)

    def check_permission(self, user_id: str, perm: Permission) -> bool:
        """Return True if the user holds *perm*."""
        user = self.get_user(user_id)
        if user is None or not user.is_active:
            return False
        role = self._roles.get(user.role_name)
        if role is None:
            return False
        return role.has_permission(perm)

    def require_permission(self, user_id: str, perm: Permission) -> None:
        """Raise :class:`PermissionError` if the user lacks *perm*."""
        if not self.check_permission(user_id, perm):
            raise PermissionError(
                f"User '{user_id}' does not have permission '{perm.value}'."
            )

    def verify_password(self, user_id: str, password: str) -> bool:
        """Return True if *password* matches the stored hash for *user_id*."""
        user = self.get_user(user_id)
        if user is None or not user._password_hash:
            return False
        parts = user._password_hash.split(":", 1)
        if len(parts) != 2:
            return False
        salt, stored_digest = parts
        digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return hmac.compare_digest(stored_digest, digest)

    @staticmethod
    def _hash_password(password: str) -> str:
        """SHA-256 hash with a random salt stored as hex."""
        salt = secrets.token_hex(16)
        digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}:{digest}"

    def assign_role(self, user_id: str, role_name: str) -> None:
        """Re-assign the role for an existing user."""
        with self._lock:
            if role_name not in self._roles:
                raise ValueError(f"Role '{role_name}' is not registered.")
            user = self._users.get(user_id)
            if user is None:
                raise KeyError(f"User '{user_id}' not found.")
            user.role_name = role_name

    def deactivate_user(self, user_id: str) -> None:
        """Deactivate a user (revokes all permissions)."""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.is_active = False


# ---------------------------------------------------------------------------
# Field-level encryption
# ---------------------------------------------------------------------------


class EncryptionManager:
    """
    AES-256-GCM field-level encryption for sensitive PII and deal data.

    Requires the ``cryptography`` package.  Keys are derived from a master
    secret via HKDF-SHA256 and a per-tenant salt so that tenants cannot
    access each other's data.

    Args:
        master_secret: 32-byte master secret.  Defaults to the
            ``SALES_LLM_MASTER_SECRET`` environment variable.
    """

    def __init__(self, master_secret: Optional[bytes] = None) -> None:
        env_secret = os.getenv("SALES_LLM_MASTER_SECRET", "")
        self._master_secret = master_secret or (
            env_secret.encode() if env_secret else secrets.token_bytes(32)
        )

    def _derive_key(self, tenant_id: str) -> bytes:
        try:
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # type: ignore
            from cryptography.hazmat.primitives import hashes  # type: ignore
        except ImportError:
            raise RuntimeError(
                "cryptography is required for encryption. "
                "Install with: pip install cryptography"
            )
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            info=b"sales-llm-encryption-v1",
        )
        return hkdf.derive(self._master_secret)

    def encrypt(self, plaintext: str, tenant_id: str) -> str:
        """
        Encrypt *plaintext* for *tenant_id*.

        Returns a Base64-encoded ``<nonce>:<ciphertext>:<tag>`` string.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
            import base64
        except ImportError:
            raise RuntimeError(
                "cryptography is required for encryption. "
                "Install with: pip install cryptography"
            )
        key = self._derive_key(tenant_id)
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
        token = base64.b64encode(nonce + ciphertext).decode()
        return token

    def decrypt(self, token: str, tenant_id: str) -> str:
        """
        Decrypt a token previously produced by :meth:`encrypt`.

        Raises:
            ValueError: If the token is invalid or has been tampered with.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
            from cryptography.exceptions import InvalidTag  # type: ignore
            import base64
        except ImportError:
            raise RuntimeError(
                "cryptography is required for encryption. "
                "Install with: pip install cryptography"
            )
        key = self._derive_key(tenant_id)
        raw = base64.b64decode(token.encode())
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise ValueError("Decryption failed: invalid tag.") from exc
        return plaintext.decode()


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


@dataclass
class AuditEvent:
    """Structured audit log entry."""

    event_id: str
    timestamp: str  # ISO-8601
    user_id: str
    tenant_id: str
    action: str  # e.g. "inference.run", "deal.read", "user.create"
    resource: str = ""  # e.g. deal ID, user email
    outcome: str = "success"  # "success" | "failure" | "denied"
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class AuditLogger:
    """
    Thread-safe structured audit logger.

    Writes JSON-formatted :class:`AuditEvent` records to a configured
    sink.  Supports:
    - In-memory buffer (default, for testing)
    - File (append mode)
    - Custom sink callable

    Args:
        sink: One of ``"memory"``, a file path string, or a callable
            ``(event: AuditEvent) -> None``.
    """

    def __init__(self, sink: Any = "memory") -> None:
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._file_path: Optional[str] = None
        self._custom_sink: Optional[Callable[[AuditEvent], None]] = None

        if callable(sink):
            self._custom_sink = sink
        elif isinstance(sink, str) and sink != "memory":
            self._file_path = sink
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(sink)), exist_ok=True)

    def log(
        self,
        user_id: str,
        tenant_id: str,
        action: str,
        resource: str = "",
        outcome: str = "success",
        detail: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Record an audit event and dispatch it to the configured sink."""
        event = AuditEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            outcome=outcome,
            detail=detail or {},
        )
        with self._lock:
            self._events.append(event)
            if self._file_path:
                with open(self._file_path, "a", encoding="utf-8") as fh:
                    fh.write(event.to_json() + "\n")
            if self._custom_sink:
                try:
                    self._custom_sink(event)
                except Exception as exc:
                    logger.error("Audit sink error: %s", exc)
        return event

    def get_events(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action_prefix: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Query in-memory audit events with optional filters."""
        with self._lock:
            events = list(self._events)
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if action_prefix:
            events = [e for e in events if e.action.startswith(action_prefix)]
        return events[-limit:]

    def clear(self) -> None:
        """Clear the in-memory event buffer (for testing)."""
        with self._lock:
            self._events.clear()


# ---------------------------------------------------------------------------
# Tenant data isolation
# ---------------------------------------------------------------------------


class TenantIsolationMiddleware:
    """
    Ensures that all data queries are scoped to a single tenant.

    Use as a context helper when building repository / service layers.

    Usage::

        with TenantIsolationMiddleware(tenant_id="acme") as ctx:
            deals = ctx.scope(raw_deals)

    Args:
        tenant_id: The identifier for the current tenant.
        tenant_field: The field name used for tenant scoping (default ``"tenant_id"``).
    """

    def __init__(self, tenant_id: str, tenant_field: str = "tenant_id") -> None:
        self.tenant_id = tenant_id
        self.tenant_field = tenant_field

    def __enter__(self) -> "TenantIsolationMiddleware":
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def scope(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter *records* to only those belonging to ``self.tenant_id``."""
        return [
            r for r in records
            if r.get(self.tenant_field) == self.tenant_id
        ]

    def tag(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add the tenant_id field to *record* (mutates in place, returns record)."""
        record[self.tenant_field] = self.tenant_id
        return record
