"""
CRM connectors for Salesforce, HubSpot, and Zoho CRM.

Each connector exposes a uniform interface so that the sales LLM can
ingest deal, contact, and activity data from the enterprise's CRM of
choice without code changes upstream.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Deal:
    """Normalised deal record surfaced to the sales LLM."""

    deal_id: str
    name: str
    stage: str
    amount: float
    currency: str = "INR"
    probability: float = 0.0
    owner_email: str = ""
    account_name: str = ""
    close_date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Contact:
    """Normalised contact record."""

    contact_id: str
    name: str
    email: str
    phone: str = ""
    title: str = ""
    account_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """Normalised CRM activity (call, email, meeting, task)."""

    activity_id: str
    type: str  # "call" | "email" | "meeting" | "task"
    subject: str
    body: str = ""
    created_at: str = ""
    owner_email: str = ""
    related_deal_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseCRMConnector(ABC):
    """Abstract base class for all CRM connectors."""

    name: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._authenticated = False

    @abstractmethod
    def authenticate(self) -> None:
        """Establish an authenticated session with the CRM."""

    @abstractmethod
    def get_deals(self, limit: int = 100, offset: int = 0) -> List[Deal]:
        """Retrieve paginated deals."""

    @abstractmethod
    def get_contacts(self, limit: int = 100, offset: int = 0) -> List[Contact]:
        """Retrieve paginated contacts."""

    @abstractmethod
    def get_activities(
        self,
        deal_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Activity]:
        """Retrieve paginated activities, optionally filtered by deal."""

    def get_all_deals(self) -> List[Deal]:
        """Convenience wrapper – fetches all deals via pagination."""
        deals, offset, batch = [], 0, 100
        while True:
            page = self.get_deals(limit=batch, offset=offset)
            deals.extend(page)
            if len(page) < batch:
                break
            offset += batch
        return deals

    def get_all_contacts(self) -> List[Contact]:
        """Convenience wrapper – fetches all contacts via pagination."""
        contacts, offset, batch = [], 0, 100
        while True:
            page = self.get_contacts(limit=batch, offset=offset)
            contacts.extend(page)
            if len(page) < batch:
                break
            offset += batch
        return contacts


# ---------------------------------------------------------------------------
# Salesforce connector
# ---------------------------------------------------------------------------


class SalesforceConnector(BaseCRMConnector):
    """
    Salesforce CRM connector via the simple-salesforce library.

    Expected config keys:
        username, password, security_token, domain (optional, default "login")
    """

    name = "salesforce"

    def authenticate(self) -> None:
        try:
            from simple_salesforce import Salesforce  # type: ignore

            self._sf = Salesforce(
                username=self.config["username"],
                password=self.config["password"],
                security_token=self.config["security_token"],
                domain=self.config.get("domain", "login"),
            )
            self._authenticated = True
            logger.info("Salesforce authentication successful.")
        except ImportError:
            raise RuntimeError(
                "simple-salesforce is required for Salesforce integration. "
                "Install it with: pip install simple-salesforce"
            )

    def get_deals(self, limit: int = 100, offset: int = 0) -> List[Deal]:
        if not self._authenticated:
            self.authenticate()
        soql = (
            f"SELECT Id, Name, StageName, Amount, CurrencyIsoCode, "
            f"Probability, Owner.Email, Account.Name, CloseDate "
            f"FROM Opportunity ORDER BY LastModifiedDate DESC "
            f"LIMIT {limit} OFFSET {offset}"
        )
        results = self._sf.query(soql)
        deals = []
        for rec in results.get("records", []):
            deals.append(
                Deal(
                    deal_id=rec["Id"],
                    name=rec.get("Name", ""),
                    stage=rec.get("StageName", ""),
                    amount=float(rec.get("Amount") or 0),
                    currency=rec.get("CurrencyIsoCode", "USD"),
                    probability=float(rec.get("Probability") or 0) / 100,
                    owner_email=(rec.get("Owner") or {}).get("Email", ""),
                    account_name=(rec.get("Account") or {}).get("Name", ""),
                    close_date=str(rec.get("CloseDate") or ""),
                )
            )
        return deals

    def get_contacts(self, limit: int = 100, offset: int = 0) -> List[Contact]:
        if not self._authenticated:
            self.authenticate()
        soql = (
            f"SELECT Id, Name, Email, Phone, Title, Account.Name "
            f"FROM Contact ORDER BY LastModifiedDate DESC "
            f"LIMIT {limit} OFFSET {offset}"
        )
        results = self._sf.query(soql)
        contacts = []
        for rec in results.get("records", []):
            contacts.append(
                Contact(
                    contact_id=rec["Id"],
                    name=rec.get("Name", ""),
                    email=rec.get("Email", ""),
                    phone=rec.get("Phone", ""),
                    title=rec.get("Title", ""),
                    account_name=(rec.get("Account") or {}).get("Name", ""),
                )
            )
        return contacts

    def get_activities(
        self,
        deal_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Activity]:
        if not self._authenticated:
            self.authenticate()
        where = f"WHERE WhatId = '{deal_id}'" if deal_id else ""
        soql = (
            f"SELECT Id, Type, Subject, Description, ActivityDate, Owner.Email "
            f"FROM Activity {where} ORDER BY ActivityDate DESC "
            f"LIMIT {limit} OFFSET {offset}"
        )
        results = self._sf.query(soql)
        activities = []
        for rec in results.get("records", []):
            activities.append(
                Activity(
                    activity_id=rec["Id"],
                    type=rec.get("Type", "task").lower(),
                    subject=rec.get("Subject", ""),
                    body=rec.get("Description", ""),
                    created_at=str(rec.get("ActivityDate") or ""),
                    owner_email=(rec.get("Owner") or {}).get("Email", ""),
                    related_deal_id=deal_id,
                )
            )
        return activities


# ---------------------------------------------------------------------------
# HubSpot connector
# ---------------------------------------------------------------------------


class HubSpotConnector(BaseCRMConnector):
    """
    HubSpot CRM connector via the hubspot-api-client library.

    Expected config keys:
        access_token
    """

    name = "hubspot"

    def authenticate(self) -> None:
        try:
            import hubspot  # type: ignore
            from hubspot import HubSpot  # type: ignore

            self._client = HubSpot(access_token=self.config["access_token"])
            self._authenticated = True
            logger.info("HubSpot authentication successful.")
        except ImportError:
            raise RuntimeError(
                "hubspot-api-client is required for HubSpot integration. "
                "Install it with: pip install hubspot-api-client"
            )

    def get_deals(self, limit: int = 100, offset: int = 0) -> List[Deal]:
        if not self._authenticated:
            self.authenticate()
        from hubspot.crm.deals import ApiException  # type: ignore

        try:
            resp = self._client.crm.deals.basic_api.get_page(
                limit=limit,
                after=str(offset) if offset else None,
                properties=["dealname", "dealstage", "amount", "closedate", "hs_probability"],
            )
        except ApiException as exc:
            logger.error("HubSpot get_deals error: %s", exc)
            return []
        deals = []
        for obj in resp.results:
            props = obj.properties or {}
            deals.append(
                Deal(
                    deal_id=obj.id,
                    name=props.get("dealname", ""),
                    stage=props.get("dealstage", ""),
                    amount=float(props.get("amount") or 0),
                    probability=float(props.get("hs_probability") or 0) / 100,
                    close_date=props.get("closedate", ""),
                )
            )
        return deals

    def get_contacts(self, limit: int = 100, offset: int = 0) -> List[Contact]:
        if not self._authenticated:
            self.authenticate()
        from hubspot.crm.contacts import ApiException  # type: ignore

        try:
            resp = self._client.crm.contacts.basic_api.get_page(
                limit=limit,
                after=str(offset) if offset else None,
                properties=["firstname", "lastname", "email", "phone", "jobtitle", "company"],
            )
        except ApiException as exc:
            logger.error("HubSpot get_contacts error: %s", exc)
            return []
        contacts = []
        for obj in resp.results:
            props = obj.properties or {}
            full_name = f"{props.get('firstname', '')} {props.get('lastname', '')}".strip()
            contacts.append(
                Contact(
                    contact_id=obj.id,
                    name=full_name,
                    email=props.get("email", ""),
                    phone=props.get("phone", ""),
                    title=props.get("jobtitle", ""),
                    account_name=props.get("company", ""),
                )
            )
        return contacts

    def get_activities(
        self,
        deal_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Activity]:
        if not self._authenticated:
            self.authenticate()
        from hubspot.crm.deals import ApiException  # type: ignore

        try:
            if deal_id:
                resp = self._client.crm.deals.associations_api.get_all(
                    deal_id=deal_id,
                    to_object_type="engagements",
                )
                engagement_ids = [a.id for a in (resp.results or [])]
            else:
                engagement_ids = []
        except (ApiException, Exception) as exc:
            logger.error("HubSpot get_activities error: %s", exc)
            return []
        activities = []
        for eid in engagement_ids[:limit]:
            activities.append(
                Activity(
                    activity_id=eid,
                    type="engagement",
                    subject="HubSpot engagement",
                    related_deal_id=deal_id,
                )
            )
        return activities


# ---------------------------------------------------------------------------
# Zoho CRM connector
# ---------------------------------------------------------------------------


class ZohoCRMConnector(BaseCRMConnector):
    """
    Zoho CRM connector via the zcrmsdk library.

    Expected config keys:
        client_id, client_secret, refresh_token, api_base_url (optional)
    """

    name = "zoho"

    def authenticate(self) -> None:
        try:
            from zcrmsdk.src.com.zoho.crm.api.initializer import Initializer  # type: ignore
            from zcrmsdk.src.com.zoho.api.authenticator.oauth_token import OAuthToken  # type: ignore
            from zcrmsdk.src.com.zoho.api.authenticator.store import DBStore  # type: ignore
            from zcrmsdk.src.com.zoho.crm.api.dc import USDataCenter  # type: ignore

            token = OAuthToken(
                client_id=self.config["client_id"],
                client_secret=self.config["client_secret"],
                refresh_token=self.config["refresh_token"],
            )
            Initializer.initialize(
                environment=USDataCenter.PRODUCTION(),
                token=token,
                store=DBStore(),
                sdk_config=None,
            )
            self._authenticated = True
            logger.info("Zoho CRM authentication successful.")
        except ImportError:
            raise RuntimeError(
                "zcrmsdk is required for Zoho CRM integration. "
                "Install it with: pip install zcrmsdk"
            )

    def get_deals(self, limit: int = 100, offset: int = 0) -> List[Deal]:
        if not self._authenticated:
            self.authenticate()
        try:
            from zcrmsdk.src.com.zoho.crm.api.deals.deals_operations import DealsOperations  # type: ignore
            from zcrmsdk.src.com.zoho.crm.api.parameter_map import ParameterMap  # type: ignore

            params = ParameterMap()
            params.add("per_page", limit)
            params.add("page", (offset // limit) + 1)
            op = DealsOperations()
            resp = op.get_deals(params)
            deals = []
            if resp and resp.get_status_code() == 200:
                for rec in resp.get_object().get_data():
                    deals.append(
                        Deal(
                            deal_id=str(rec.get_id()),
                            name=rec.get_deal_name() or "",
                            stage=rec.get_stage() or "",
                            amount=float(rec.get_amount() or 0),
                            close_date=str(rec.get_closing_date() or ""),
                        )
                    )
            return deals
        except Exception as exc:
            logger.error("Zoho CRM get_deals error: %s", exc)
            return []

    def get_contacts(self, limit: int = 100, offset: int = 0) -> List[Contact]:
        if not self._authenticated:
            self.authenticate()
        try:
            from zcrmsdk.src.com.zoho.crm.api.contacts.contacts_operations import ContactsOperations  # type: ignore
            from zcrmsdk.src.com.zoho.crm.api.parameter_map import ParameterMap  # type: ignore

            params = ParameterMap()
            params.add("per_page", limit)
            params.add("page", (offset // limit) + 1)
            op = ContactsOperations()
            resp = op.get_contacts(params)
            contacts = []
            if resp and resp.get_status_code() == 200:
                for rec in resp.get_object().get_data():
                    contacts.append(
                        Contact(
                            contact_id=str(rec.get_id()),
                            name=f"{rec.get_first_name() or ''} {rec.get_last_name() or ''}".strip(),
                            email=rec.get_email() or "",
                            phone=rec.get_phone() or "",
                            title=rec.get_title() or "",
                            account_name=str(rec.get_account_name() or ""),
                        )
                    )
            return contacts
        except Exception as exc:
            logger.error("Zoho CRM get_contacts error: %s", exc)
            return []

    def get_activities(
        self,
        deal_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Activity]:
        if not self._authenticated:
            self.authenticate()
        try:
            from zcrmsdk.src.com.zoho.crm.api.activities.activities_operations import ActivitiesOperations  # type: ignore
            from zcrmsdk.src.com.zoho.crm.api.parameter_map import ParameterMap  # type: ignore

            params = ParameterMap()
            params.add("per_page", limit)
            params.add("page", (offset // limit) + 1)
            op = ActivitiesOperations()
            resp = op.get_activities(params)
            activities = []
            if resp and resp.get_status_code() == 200:
                for rec in resp.get_object().get_data():
                    activities.append(
                        Activity(
                            activity_id=str(rec.get_id()),
                            type=(rec.get_activity_type() or "task").lower(),
                            subject=rec.get_subject() or "",
                        )
                    )
            return activities
        except Exception as exc:
            logger.error("Zoho CRM get_activities error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CONNECTOR_REGISTRY: Dict[str, type] = {
    "salesforce": SalesforceConnector,
    "hubspot": HubSpotConnector,
    "zoho": ZohoCRMConnector,
}


def get_crm_connector(name: str, config: Dict[str, Any]) -> BaseCRMConnector:
    """Return an instantiated CRM connector by name.

    Args:
        name: One of ``"salesforce"``, ``"hubspot"``, ``"zoho"``.
        config: Connector-specific authentication configuration.

    Raises:
        ValueError: If *name* is not a registered connector.
    """
    name = name.lower()
    if name not in _CONNECTOR_REGISTRY:
        raise ValueError(
            f"Unknown CRM connector '{name}'. "
            f"Available: {list(_CONNECTOR_REGISTRY)}"
        )
    return _CONNECTOR_REGISTRY[name](config)
