"""
Unit tests for the enterprise sales LLM module.

These tests cover:
- Configuration (SalesLLMConfig)
- CRM connector factory
- Communication connector factory
- ETL pipeline extractors
- Sales intelligence (lead scoring, deal prediction, churn, NBA, forecasting)
- Security (RBAC, encryption, audit logging, tenant isolation)
- Deployment helpers (Docker Compose and Kubernetes manifest generation)
"""

import io
import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSalesLLMConfig:
    def test_default_config(self):
        from h2ogpt.src.sales.config import SalesLLMConfig, ModelSize, ContextLength

        cfg = SalesLLMConfig()
        assert cfg.model.model_size == ModelSize.B_70
        assert cfg.model.context_length == ContextLength.K_128
        assert cfg.model.gpu_enabled is True
        assert cfg.model.cpu_fallback is True
        assert cfg.model.streaming is True
        assert cfg.model.target_latency_seconds == 1.5
        assert "salesforce" in cfg.crm_connectors
        assert "hubspot" in cfg.crm_connectors
        assert "zoho" in cfg.crm_connectors
        assert "gmail" in cfg.communication_connectors
        assert "slack" in cfg.communication_connectors
        assert "csv" in cfg.etl_sources
        assert "pdf" in cfg.etl_sources
        assert cfg.security_enabled is True
        assert cfg.intelligence_enabled is True

    def test_from_env(self, monkeypatch):
        from h2ogpt.src.sales.config import SalesLLMConfig, ModelSize, DeploymentMode

        monkeypatch.setenv("SALES_LLM_MODEL_SIZE", "30b")
        monkeypatch.setenv("SALES_LLM_DEPLOYMENT_MODE", "kubernetes")
        monkeypatch.setenv("SALES_LLM_API_PORT", "9000")
        monkeypatch.setenv("SALES_LLM_STREAMING", "false")

        cfg = SalesLLMConfig.from_env()
        assert cfg.model.model_size == ModelSize.B_30
        assert cfg.deployment_mode == DeploymentMode.KUBERNETES
        assert cfg.api_port == 9000
        assert cfg.model.streaming is False

    def test_model_config_fields(self):
        from h2ogpt.src.sales.config import ModelConfig

        m = ModelConfig(quantization="int4", num_gpus=2)
        assert m.quantization == "int4"
        assert m.num_gpus == 2

    def test_rag_config_defaults(self):
        from h2ogpt.src.sales.config import RAGConfig

        r = RAGConfig()
        assert r.enabled is True
        assert r.reranking_enabled is True
        assert r.top_k == 10
        assert r.vector_db == "chroma"


# ---------------------------------------------------------------------------
# CRM connector tests
# ---------------------------------------------------------------------------


class TestCRMConnectors:
    def test_factory_unknown_raises(self):
        from h2ogpt.src.sales.crm_connectors import get_crm_connector

        with pytest.raises(ValueError, match="Unknown CRM connector"):
            get_crm_connector("unknown_crm", {})

    def test_factory_salesforce(self):
        from h2ogpt.src.sales.crm_connectors import get_crm_connector, SalesforceConnector

        connector = get_crm_connector("salesforce", {"username": "u"})
        assert isinstance(connector, SalesforceConnector)

    def test_factory_hubspot(self):
        from h2ogpt.src.sales.crm_connectors import get_crm_connector, HubSpotConnector

        connector = get_crm_connector("hubspot", {"access_token": "t"})
        assert isinstance(connector, HubSpotConnector)

    def test_factory_zoho(self):
        from h2ogpt.src.sales.crm_connectors import get_crm_connector, ZohoCRMConnector

        connector = get_crm_connector("zoho", {"client_id": "c"})
        assert isinstance(connector, ZohoCRMConnector)

    def test_deal_dataclass(self):
        from h2ogpt.src.sales.crm_connectors import Deal

        d = Deal(deal_id="123", name="Acme Corp", stage="Proposal", amount=5_000_000)
        assert d.currency == "INR"
        assert d.amount == 5_000_000
        assert d.extra == {}

    def test_contact_dataclass(self):
        from h2ogpt.src.sales.crm_connectors import Contact

        c = Contact(contact_id="c1", name="Alice", email="alice@acme.com")
        assert c.phone == ""
        assert c.account_name == ""

    def test_activity_dataclass(self):
        from h2ogpt.src.sales.crm_connectors import Activity

        a = Activity(activity_id="a1", type="call", subject="Discovery call")
        assert a.body == ""
        assert a.related_deal_id is None

    def test_connector_not_authenticated_initially(self):
        from h2ogpt.src.sales.crm_connectors import SalesforceConnector

        c = SalesforceConnector({})
        assert c._authenticated is False

    def test_get_all_deals_stops_when_page_short(self):
        """get_all_deals should stop pagination when a short page is returned."""
        from h2ogpt.src.sales.crm_connectors import BaseCRMConnector, Deal

        class MockCRM(BaseCRMConnector):
            def authenticate(self):
                self._authenticated = True

            def get_deals(self, limit=100, offset=0):
                if offset == 0:
                    return [Deal("1", "D1", "Proposal", 1.0)] * 5
                return []

            def get_contacts(self, limit=100, offset=0):
                return []

            def get_activities(self, deal_id=None, limit=100, offset=0):
                return []

        crm = MockCRM({})
        crm.authenticate()
        deals = crm.get_all_deals()
        assert len(deals) == 5


# ---------------------------------------------------------------------------
# Communication connector tests
# ---------------------------------------------------------------------------


class TestCommunicationConnectors:
    def test_factory_unknown_raises(self):
        from h2ogpt.src.sales.communication_connectors import get_communication_connector

        with pytest.raises(ValueError, match="Unknown communication connector"):
            get_communication_connector("telegram", {})

    def test_factory_all_platforms(self):
        from h2ogpt.src.sales.communication_connectors import (
            get_communication_connector,
            GmailConnector,
            OutlookConnector,
            SlackConnector,
            WhatsAppBusinessConnector,
            ZoomConnector,
            TeamsConnector,
        )

        pairs = [
            ("gmail", GmailConnector),
            ("outlook", OutlookConnector),
            ("slack", SlackConnector),
            ("whatsapp", WhatsAppBusinessConnector),
            ("zoom", ZoomConnector),
            ("teams", TeamsConnector),
        ]
        for name, cls in pairs:
            c = get_communication_connector(name, {"dummy": "config"})
            assert isinstance(c, cls), f"Expected {cls.__name__} for '{name}'"

    def test_message_dataclass(self):
        from h2ogpt.src.sales.communication_connectors import Message

        m = Message(
            message_id="m1",
            platform="gmail",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            subject="Q4 Deal",
            body="Let's discuss the deal.",
        )
        assert m.platform == "gmail"
        assert m.thread_id is None
        assert m.attachments == []


# ---------------------------------------------------------------------------
# ETL pipeline tests
# ---------------------------------------------------------------------------


class TestETLPipeline:
    def test_csv_extractor(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import CSVExtractor

        csv_file = tmp_path / "deals.csv"
        csv_file.write_text("deal_id,name,amount\n001,Acme Corp,5000000\n002,Beta Inc,2000000\n")
        extractor = CSVExtractor(paths=[csv_file], id_column="deal_id")
        docs = extractor.extract()
        assert len(docs) == 2
        assert docs[0].doc_id == "001"
        assert "Acme Corp" in docs[0].content
        assert docs[0].metadata["deal_id"] == "001"

    def test_csv_extractor_fallback_id(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import CSVExtractor

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nfoo,1\nbar,2\n")
        extractor = CSVExtractor(paths=[csv_file])
        docs = extractor.extract()
        assert len(docs) == 2
        assert docs[0].doc_id.endswith("row0")

    def test_email_extractor_raw(self):
        from h2ogpt.src.sales.etl_pipeline import EmailExtractor

        raw_email = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Follow-up on proposal\r\n"
            "Date: Mon, 01 Jan 2024 12:00:00 +0000\r\n"
            "Content-Type: text/plain\r\n"
            "\r\n"
            "Hi Bob, just following up on the proposal.\r\n"
        )
        extractor = EmailExtractor(sources=[raw_email])
        docs = extractor.extract()
        assert len(docs) == 1
        assert "alice@example.com" in docs[0].content
        assert "Follow-up on proposal" in docs[0].content
        assert docs[0].metadata["subject"] == "Follow-up on proposal"

    def test_email_extractor_file(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import EmailExtractor

        eml = tmp_path / "sample.eml"
        eml.write_text(
            "From: rep@company.com\r\nSubject: Deal update\r\nContent-Type: text/plain\r\n\r\nDeal is progressing.\r\n"
        )
        extractor = EmailExtractor(sources=[eml])
        docs = extractor.extract()
        assert len(docs) == 1
        assert "rep@company.com" in docs[0].content

    def test_call_transcript_extractor_plain_text(self):
        from h2ogpt.src.sales.etl_pipeline import CallTranscriptExtractor

        transcript = "Sales Rep: Tell me about your challenges.\nProspect: We need a better CRM integration.\n"
        extractor = CallTranscriptExtractor(sources=[transcript])
        docs = extractor.extract()
        assert len(docs) >= 1
        assert "CRM integration" in docs[0].content

    def test_call_transcript_extractor_vtt(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import CallTranscriptExtractor

        vtt = tmp_path / "call.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello, welcome to the demo.\n\n"
            "00:00:04.000 --> 00:00:07.000\nThank you for joining.\n"
        )
        extractor = CallTranscriptExtractor(sources=[vtt])
        docs = extractor.extract()
        assert len(docs) >= 1
        joined = " ".join(d.content for d in docs)
        assert "Hello" in joined or "Thank you" in joined

    def test_call_transcript_strip_vtt(self):
        from h2ogpt.src.sales.etl_pipeline import CallTranscriptExtractor

        raw_vtt = "WEBVTT\n\n1\n00:00:00,000 --> 00:00:02,000\nThis is a test transcript.\n"
        text = CallTranscriptExtractor._strip_vtt_srt(raw_vtt)
        assert "00:00:00" not in text
        assert "This is a test transcript." in text

    def test_etl_pipeline_run_and_dedup(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import ETLPipeline, CSVExtractor, EmailExtractor

        csv1 = tmp_path / "a.csv"
        csv1.write_text("id,name\n1,Alpha\n2,Beta\n")
        raw_email = "From: x@y.com\r\nSubject: Hi\r\n\r\nBody.\r\n"

        pipeline = ETLPipeline()
        pipeline.add_extractor(CSVExtractor([csv1]))
        pipeline.add_extractor(EmailExtractor([raw_email]))
        # Run twice to test dedup
        pipeline.add_extractor(CSVExtractor([csv1]))

        docs = pipeline.run()
        # After dedup: 2 CSV rows + 1 email = 3 unique docs
        assert len(docs) == 3

    def test_etl_pipeline_chaining(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import ETLPipeline, CSVExtractor

        csv_file = tmp_path / "leads.csv"
        csv_file.write_text("id,company\n10,Gamma Corp\n20,Delta Ltd\n")

        docs = (
            ETLPipeline()
            .add_extractor(CSVExtractor([csv_file]))
            .run()
        )
        assert len(docs) == 2

    def test_build_pipeline_from_config(self, tmp_path):
        from h2ogpt.src.sales.etl_pipeline import build_pipeline_from_config

        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text("cid,name\nc1,Alice\nc2,Bob\n")
        config = {
            "csv": [{"paths": [str(csv_file)], "id_column": "cid"}],
        }
        pipeline = build_pipeline_from_config(config)
        docs = pipeline.run()
        assert len(docs) == 2
        assert docs[0].doc_id in ("c1", "c2")

    def test_document_dataclass(self):
        from h2ogpt.src.sales.etl_pipeline import Document

        d = Document(doc_id="d1", source="test.csv", content="deal: Acme")
        assert d.metadata == {}
        assert d.doc_id == "d1"


# ---------------------------------------------------------------------------
# Sales intelligence tests
# ---------------------------------------------------------------------------


class TestSalesIntelligence:
    def _make_engine(self):
        from h2ogpt.src.sales.intelligence import SalesIntelligence
        return SalesIntelligence()

    def test_score_lead_grade_a(self):
        eng = self._make_engine()
        lead = {
            "lead_id": "L001",
            "name": "BigCorp",
            "company_size": 2000,
            "budget": 25_000_000,
            "engagement_score": 10,
            "icp_match": 1,
            "days_to_decision": 30,
        }
        result = eng.score_lead(lead)
        assert result.grade == "A"
        assert result.score >= 80

    def test_score_lead_grade_d(self):
        eng = self._make_engine()
        lead = {
            "lead_id": "L002",
            "company_size": 5,
            "budget": 100_000,
            "engagement_score": 0,
            "icp_match": 0,
            "days_to_decision": 365,
        }
        result = eng.score_lead(lead)
        assert result.grade == "D"
        assert result.score < 40

    def test_score_lead_factors_present(self):
        eng = self._make_engine()
        lead = {"lead_id": "L003", "company_size": 500, "budget": 10_000_000}
        result = eng.score_lead(lead)
        assert "company_size" in result.factors
        assert "budget_range" in result.factors
        assert result.reasoning != ""

    def test_score_lead_with_llm(self):
        from h2ogpt.src.sales.intelligence import SalesIntelligence

        eng = SalesIntelligence(llm_client=lambda prompt: "LLM reasoning output.")
        lead = {"lead_id": "L004", "company_size": 1000, "budget": 15_000_000, "icp_match": 1}
        result = eng.score_lead(lead)
        assert result.reasoning == "LLM reasoning output."

    def test_predict_deal_closure_won(self):
        eng = self._make_engine()
        deal = {"deal_id": "D001", "stage": "closed won"}
        result = eng.predict_deal_closure(deal)
        assert result.close_probability == 1.0

    def test_predict_deal_closure_lost(self):
        eng = self._make_engine()
        deal = {"deal_id": "D002", "stage": "closed lost"}
        result = eng.predict_deal_closure(deal)
        assert result.close_probability == 0.0

    def test_predict_deal_closure_risk_flags(self):
        eng = self._make_engine()
        deal = {
            "deal_id": "D003",
            "stage": "negotiation/review",
            "days_since_last_activity": 45,
            "stakeholders_count": 1,
        }
        result = eng.predict_deal_closure(deal)
        assert len(result.risk_flags) > 0
        assert any("activity" in f.lower() for f in result.risk_flags)

    def test_predict_deal_closure_crm_prob_override(self):
        eng = self._make_engine()
        deal = {"deal_id": "D004", "stage": "prospecting", "crm_probability": 0.95}
        result = eng.predict_deal_closure(deal)
        # CRM probability (0.95) overrides base stage weight (0.10)
        assert result.close_probability >= 0.80

    def test_detect_churn_low_risk(self):
        eng = self._make_engine()
        account = {
            "account_id": "A001",
            "account_name": "Happy Corp",
            "nps_score": 9,
            "support_tickets_30d": 1,
            "usage_change_pct": 10,
            "days_to_renewal": 180,
            "payment_delay_days": 0,
        }
        result = eng.detect_churn(account)
        assert result.risk_level == "low"
        assert result.churn_probability < 0.25

    def test_detect_churn_critical(self):
        eng = self._make_engine()
        account = {
            "account_id": "A002",
            "account_name": "At-Risk Ltd",
            "nps_score": 2,
            "support_tickets_30d": 20,
            "usage_change_pct": -50,
            "days_to_renewal": 10,
            "payment_delay_days": 45,
        }
        result = eng.detect_churn(account)
        assert result.risk_level == "critical"
        assert result.churn_probability >= 0.70

    def test_detect_churn_reasons_populated(self):
        eng = self._make_engine()
        account = {
            "account_id": "A003",
            "account_name": "Risky Corp",
            "nps_score": 4,
            "support_tickets_30d": 15,
        }
        result = eng.detect_churn(account)
        assert len(result.reasons) > 0

    def test_recommend_next_action_idle_deal(self):
        eng = self._make_engine()
        entity = {"deal_id": "D010", "stage": "qualification", "days_since_last_activity": 20}
        result = eng.recommend_next_action(entity, entity_type="deal")
        assert result.entity_id == "D010"
        assert result.priority <= 3
        assert result.action != ""

    def test_recommend_next_action_llm(self):
        from h2ogpt.src.sales.intelligence import SalesIntelligence

        eng = SalesIntelligence(llm_client=lambda p: "Schedule a demo call.")
        entity = {"deal_id": "D011", "stage": "proposal"}
        result = eng.recommend_next_action(entity)
        assert result.action == "Schedule a demo call."

    def test_draft_follow_up_template(self):
        eng = self._make_engine()
        deal = {
            "deal_id": "D020",
            "name": "Mega Corp Deal",
            "contact_name": "John",
            "stage": "Proposal",
        }
        draft = eng.draft_follow_up(deal)
        assert draft.channel == "email"
        assert "John" in draft.body
        assert "Mega Corp Deal" in draft.subject or "Mega Corp Deal" in draft.body

    def test_draft_follow_up_llm(self):
        from h2ogpt.src.sales.intelligence import SalesIntelligence

        eng = SalesIntelligence(llm_client=lambda p: "Subject: Next Steps\n\nHi, let's talk.")
        deal = {"deal_id": "D021", "name": "X Corp", "stage": "Negotiation"}
        draft = eng.draft_follow_up(deal, channel="slack")
        assert "Next Steps" in draft.subject
        assert "let's talk" in draft.body

    def test_handle_objection_price(self):
        eng = self._make_engine()
        response = eng.handle_objection("Your price is too expensive for our budget.")
        assert len(response) > 20
        assert any(kw in response.lower() for kw in ["roi", "cost", "budget", "payment"])

    def test_handle_objection_time(self):
        eng = self._make_engine()
        response = eng.handle_objection("We are too busy to evaluate this right now.")
        assert "time" in response.lower() or "minutes" in response.lower() or "hours" in response.lower()

    def test_handle_objection_llm(self):
        from h2ogpt.src.sales.intelligence import SalesIntelligence

        eng = SalesIntelligence(llm_client=lambda p: "Understood, let me address that.")
        response = eng.handle_objection("We already have a vendor.")
        assert response == "Understood, let me address that."

    def test_handle_objection_generic(self):
        eng = self._make_engine()
        response = eng.handle_objection("I have no specific objection right now.")
        assert len(response) > 0

    def test_forecast_revenue_basic(self):
        eng = self._make_engine()
        deals = [
            {"deal_id": "D100", "name": "Deal A", "amount": 10_000_000, "crm_probability": 0.8, "stage": "negotiation/review"},
            {"deal_id": "D101", "name": "Deal B", "amount": 5_000_000, "crm_probability": 0.4, "stage": "qualification"},
            {"deal_id": "D102", "name": "Deal C", "amount": 3_000_000, "crm_probability": 1.0, "stage": "closed won"},
        ]
        forecast = eng.forecast_revenue(deals, period="2024-Q4")
        assert forecast.period == "2024-Q4"
        # expected = 10M*0.8 + 5M*0.4 + 3M*1.0 = 8M + 2M + 3M = 13M
        assert forecast.expected_revenue == pytest.approx(13_000_000.0, rel=0.01)
        assert forecast.best_case_revenue >= 13_000_000.0  # deals with prob ≥0.7 + closed won
        assert forecast.worst_case_revenue == 3_000_000.0  # only closed won
        assert len(forecast.deal_breakdown) == 3

    def test_forecast_revenue_empty(self):
        eng = self._make_engine()
        forecast = eng.forecast_revenue([])
        assert forecast.expected_revenue == 0.0
        assert forecast.best_case_revenue == 0.0
        assert forecast.worst_case_revenue == 0.0
        assert forecast.deal_breakdown == []

    def test_forecast_revenue_currency(self):
        eng = self._make_engine()
        forecast = eng.forecast_revenue([], currency="USD")
        assert forecast.currency == "USD"


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


class TestRBACManager:
    def _make_manager(self):
        from h2ogpt.src.sales.security import RBACManager
        return RBACManager()

    def test_create_user_and_check_permission(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("u1", "alice@acme.com", "tenant1", "sales_rep")
        assert mgr.check_permission("u1", Permission.READ_DEALS)
        assert mgr.check_permission("u1", Permission.RUN_INFERENCE)
        assert not mgr.check_permission("u1", Permission.MANAGE_USERS)

    def test_admin_has_all_permissions(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("admin1", "admin@acme.com", "tenant1", "admin")
        for perm in Permission:
            assert mgr.check_permission("admin1", perm)

    def test_require_permission_raises(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("u2", "bob@acme.com", "tenant1", "read_only")
        with pytest.raises(PermissionError):
            mgr.require_permission("u2", Permission.WRITE_DEALS)

    def test_require_permission_passes(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("u3", "carol@acme.com", "tenant1", "sales_manager")
        mgr.require_permission("u3", Permission.USE_INTELLIGENCE)  # should not raise

    def test_deactivated_user_has_no_permissions(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("u4", "dave@acme.com", "tenant1", "admin")
        mgr.deactivate_user("u4")
        assert not mgr.check_permission("u4", Permission.READ_DEALS)

    def test_unknown_user_has_no_permissions(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        assert not mgr.check_permission("nonexistent", Permission.READ_DEALS)

    def test_duplicate_user_raises(self):
        mgr = self._make_manager()
        mgr.create_user("u5", "eve@acme.com", "tenant1", "sales_rep")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_user("u5", "eve2@acme.com", "tenant1", "sales_rep")

    def test_unknown_role_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="not registered"):
            mgr.create_user("u6", "frank@acme.com", "tenant1", "nonexistent_role")

    def test_assign_role(self):
        from h2ogpt.src.sales.security import Permission

        mgr = self._make_manager()
        mgr.create_user("u7", "grace@acme.com", "tenant1", "read_only")
        assert not mgr.check_permission("u7", Permission.WRITE_DEALS)
        mgr.assign_role("u7", "sales_rep")
        assert mgr.check_permission("u7", Permission.WRITE_DEALS)

    def test_verify_password(self):
        mgr = self._make_manager()
        mgr.create_user("u8", "henry@acme.com", "tenant1", "sales_rep", password="s3cr3t!")
        assert mgr.verify_password("u8", "s3cr3t!")
        assert not mgr.verify_password("u8", "wrong_password")

    def test_custom_role(self):
        from h2ogpt.src.sales.security import RBACManager, Role, Permission

        custom = Role(
            name="forecaster",
            permissions=frozenset({Permission.READ_FORECASTS, Permission.READ_DEALS}),
        )
        mgr = RBACManager(custom_roles=[custom])
        mgr.create_user("u9", "iris@acme.com", "tenant1", "forecaster")
        assert mgr.check_permission("u9", Permission.READ_FORECASTS)
        assert not mgr.check_permission("u9", Permission.WRITE_DEALS)


class TestEncryptionManager:
    def test_encrypt_decrypt_roundtrip(self):
        from h2ogpt.src.sales.security import EncryptionManager

        try:
            mgr = EncryptionManager(master_secret=b"a" * 32)
            plaintext = "Acme Corp deal value: ₹5,00,00,000"
            token = mgr.encrypt(plaintext, tenant_id="tenant_acme")
            recovered = mgr.decrypt(token, tenant_id="tenant_acme")
            assert recovered == plaintext
        except RuntimeError as exc:
            if "cryptography" in str(exc):
                pytest.skip("cryptography package not installed")
            raise

    def test_wrong_tenant_cannot_decrypt(self):
        from h2ogpt.src.sales.security import EncryptionManager

        try:
            mgr = EncryptionManager(master_secret=b"b" * 32)
            token = mgr.encrypt("secret", tenant_id="tenant_a")
            with pytest.raises(ValueError, match="Decryption failed"):
                mgr.decrypt(token, tenant_id="tenant_b")
        except RuntimeError as exc:
            if "cryptography" in str(exc):
                pytest.skip("cryptography package not installed")
            raise

    def test_different_values_produce_different_tokens(self):
        from h2ogpt.src.sales.security import EncryptionManager

        try:
            mgr = EncryptionManager(master_secret=b"c" * 32)
            t1 = mgr.encrypt("value_one", tenant_id="t1")
            t2 = mgr.encrypt("value_one", tenant_id="t1")
            # Nonces differ, so tokens must differ
            assert t1 != t2
        except RuntimeError as exc:
            if "cryptography" in str(exc):
                pytest.skip("cryptography package not installed")
            raise


class TestAuditLogger:
    def test_log_event(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        event = al.log("u1", "tenant1", "deal.read", resource="D001")
        assert event.action == "deal.read"
        assert event.user_id == "u1"
        assert event.outcome == "success"
        assert event.event_id != ""

    def test_get_events_filter_by_tenant(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        al.log("u1", "tenant_a", "deal.read")
        al.log("u2", "tenant_b", "deal.write")
        al.log("u3", "tenant_a", "inference.run")
        events = al.get_events(tenant_id="tenant_a")
        assert len(events) == 2
        assert all(e.tenant_id == "tenant_a" for e in events)

    def test_get_events_filter_by_user(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        al.log("alice", "t1", "deal.read")
        al.log("bob", "t1", "deal.write")
        al.log("alice", "t1", "inference.run")
        events = al.get_events(user_id="alice")
        assert len(events) == 2

    def test_get_events_filter_by_action_prefix(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        al.log("u1", "t1", "deal.read")
        al.log("u1", "t1", "inference.run")
        al.log("u1", "t1", "deal.write")
        events = al.get_events(action_prefix="deal.")
        assert len(events) == 2

    def test_audit_event_to_json(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        event = al.log("u1", "t1", "user.create", resource="new_user@corp.com")
        data = json.loads(event.to_json())
        assert data["action"] == "user.create"
        assert data["resource"] == "new_user@corp.com"

    def test_file_sink(self, tmp_path):
        from h2ogpt.src.sales.security import AuditLogger

        log_file = tmp_path / "audit.log"
        al = AuditLogger(sink=str(log_file))
        al.log("u1", "t1", "deal.read")
        al.log("u2", "t1", "deal.write")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["action"] == "deal.read"

    def test_clear(self):
        from h2ogpt.src.sales.security import AuditLogger

        al = AuditLogger()
        al.log("u1", "t1", "x")
        al.clear()
        assert al.get_events() == []

    def test_custom_sink(self):
        from h2ogpt.src.sales.security import AuditLogger, AuditEvent

        received = []
        al = AuditLogger(sink=received.append)
        al.log("u1", "t1", "test.action")
        assert len(received) == 1
        assert isinstance(received[0], AuditEvent)


class TestTenantIsolation:
    def test_scope_filters_correct_tenant(self):
        from h2ogpt.src.sales.security import TenantIsolationMiddleware

        records = [
            {"tenant_id": "a", "name": "Deal 1"},
            {"tenant_id": "b", "name": "Deal 2"},
            {"tenant_id": "a", "name": "Deal 3"},
        ]
        with TenantIsolationMiddleware("a") as ctx:
            scoped = ctx.scope(records)
        assert len(scoped) == 2
        assert all(r["tenant_id"] == "a" for r in scoped)

    def test_tag_adds_tenant_id(self):
        from h2ogpt.src.sales.security import TenantIsolationMiddleware

        record = {"name": "New Deal", "amount": 1_000_000}
        with TenantIsolationMiddleware("acme") as ctx:
            ctx.tag(record)
        assert record["tenant_id"] == "acme"


# ---------------------------------------------------------------------------
# Deployment tests
# ---------------------------------------------------------------------------


class TestDeploymentHelpers:
    def test_generate_docker_compose_contains_services(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_docker_compose

        cfg = SalesLLMConfig()
        yml = generate_docker_compose(cfg)
        assert "sales-llm:" in yml
        assert "chromadb:" in yml
        assert "redis:" in yml
        assert "nginx:" in yml

    def test_generate_docker_compose_uses_config_port(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_docker_compose

        cfg = SalesLLMConfig()
        cfg.api_port = 9999
        yml = generate_docker_compose(cfg)
        assert "9999" in yml

    def test_generate_docker_compose_gpu_section(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_docker_compose

        cfg = SalesLLMConfig()
        cfg.model.gpu_enabled = True
        yml = generate_docker_compose(cfg)
        assert "nvidia" in yml

    def test_generate_docker_compose_writes_file(self, tmp_path):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_docker_compose

        cfg = SalesLLMConfig()
        out = tmp_path / "compose.yml"
        generate_docker_compose(cfg, output_path=str(out))
        assert out.exists()
        content = out.read_text()
        assert "sales-llm:" in content

    def test_generate_kubernetes_manifest_contains_resources(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_kubernetes_manifest

        cfg = SalesLLMConfig()
        yml = generate_kubernetes_manifest(cfg)
        assert "Namespace" in yml
        assert "Deployment" in yml
        assert "Service" in yml
        assert "HorizontalPodAutoscaler" in yml
        assert "ConfigMap" in yml

    def test_generate_kubernetes_manifest_gpu_toleration(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_kubernetes_manifest

        cfg = SalesLLMConfig()
        cfg.model.gpu_enabled = True
        yml = generate_kubernetes_manifest(cfg)
        assert "nvidia.com/gpu" in yml
        assert "tolerations" in yml

    def test_generate_kubernetes_manifest_writes_file(self, tmp_path):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import generate_kubernetes_manifest

        cfg = SalesLLMConfig()
        out = tmp_path / "k8s.yml"
        generate_kubernetes_manifest(cfg, output_path=str(out))
        assert out.exists()

    def test_check_gpu_availability_returns_dict(self):
        from h2ogpt.src.sales.deployment import check_gpu_availability

        result = check_gpu_availability()
        assert "available" in result
        assert "device_count" in result
        assert "device_names" in result
        assert isinstance(result["device_names"], list)

    def test_get_deployment_summary(self):
        from h2ogpt.src.sales.config import SalesLLMConfig
        from h2ogpt.src.sales.deployment import get_deployment_summary

        cfg = SalesLLMConfig()
        summary = get_deployment_summary(cfg)
        assert "Sales LLM" in summary
        assert "salesforce" in summary
        assert "gmail" in summary
        assert "chroma" in summary
