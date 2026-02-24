"""
Sales intelligence module for enterprise sales LLM.

Provides deterministic and ML-assisted heuristics for:
- Lead scoring
- Deal-closure probability prediction
- Churn / at-risk account detection
- Next-best-action recommendation
- Automated follow-up drafting
- Objection handling suggestions
- Revenue forecasting
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LeadScore:
    """Lead scoring result."""

    lead_id: str
    score: float  # 0.0 – 100.0
    grade: str  # "A" | "B" | "C" | "D"
    reasoning: str = ""
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class DealPrediction:
    """Deal-closure prediction result."""

    deal_id: str
    close_probability: float  # 0.0 – 1.0
    predicted_close_date: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ChurnAlert:
    """Churn / at-risk account alert."""

    account_id: str
    account_name: str
    churn_probability: float  # 0.0 – 1.0
    risk_level: str  # "low" | "medium" | "high" | "critical"
    reasons: List[str] = field(default_factory=list)


@dataclass
class NextBestAction:
    """Next-best-action recommendation for a deal or contact."""

    entity_id: str
    entity_type: str  # "deal" | "contact" | "account"
    action: str  # short description, e.g. "Schedule discovery call"
    rationale: str = ""
    priority: int = 5  # 1 (highest) – 10 (lowest)
    due_date: Optional[str] = None


@dataclass
class FollowUpDraft:
    """Auto-generated follow-up message draft."""

    deal_id: str
    channel: str  # "email" | "slack" | "whatsapp"
    subject: str
    body: str
    suggested_send_time: Optional[str] = None


@dataclass
class RevenueForecast:
    """Revenue forecast for a time period."""

    period: str  # e.g. "2024-Q4"
    expected_revenue: float
    best_case_revenue: float
    worst_case_revenue: float
    currency: str = "INR"
    deal_breakdown: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core intelligence engine
# ---------------------------------------------------------------------------


class SalesIntelligence:
    """
    Enterprise sales intelligence engine.

    Applies rule-based heuristics and, when available, integrates with the
    sales LLM for natural-language reasoning about deals, leads, accounts,
    and revenue.

    Args:
        llm_client: Optional callable ``(prompt: str) -> str`` backed by the
            sales LLM.  When provided, qualitative analysis uses the LLM.
        config: Optional configuration overrides.
    """

    _GRADE_THRESHOLDS = {"A": 80.0, "B": 60.0, "C": 40.0, "D": 0.0}

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    # ------------------------------------------------------------------
    # Lead scoring
    # ------------------------------------------------------------------

    def score_lead(
        self,
        lead: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> LeadScore:
        """
        Score a lead from 0–100.

        Factors (default weights):
        - ``company_size``   – 0–25 pts (employees)
        - ``budget_range``   – 0–25 pts (INR)
        - ``engagement``     – 0–20 pts (email opens, calls, meetings)
        - ``fit``            – 0–15 pts (ICP match)
        - ``urgency``        – 0–15 pts (timeline to purchase)

        Args:
            lead: Dict with keys like ``company_size``, ``budget``,
                ``engagement_score``, ``icp_match``, ``days_to_decision``.
            weights: Optional override of factor weights.

        Returns:
            :class:`LeadScore` with normalised score and grade.
        """
        w = weights or {
            "company_size": 25.0,
            "budget_range": 25.0,
            "engagement": 20.0,
            "fit": 15.0,
            "urgency": 15.0,
        }
        factors: Dict[str, float] = {}

        # Company size (employees)
        emp = lead.get("company_size", 0) or 0
        size_score = min(emp / 1000, 1.0) * w["company_size"]
        factors["company_size"] = round(size_score, 2)

        # Budget (INR)
        budget = lead.get("budget", 0) or 0
        # ₹2Cr = 20_000_000 → full score
        budget_score = min(budget / 20_000_000, 1.0) * w["budget_range"]
        factors["budget_range"] = round(budget_score, 2)

        # Engagement (0–10 normalised)
        engagement_raw = lead.get("engagement_score", 0) or 0
        engagement_score = min(engagement_raw / 10, 1.0) * w["engagement"]
        factors["engagement"] = round(engagement_score, 2)

        # ICP fit (0–1 bool or float)
        icp = lead.get("icp_match", 0) or 0
        fit_score = float(icp) * w["fit"]
        factors["fit"] = round(fit_score, 2)

        # Urgency: days to decision (lower = more urgent = higher score)
        days = lead.get("days_to_decision", 365) or 365
        urgency_score = max(0.0, 1.0 - days / 365) * w["urgency"]
        factors["urgency"] = round(urgency_score, 2)

        total = sum(factors.values())
        grade = "D"
        for g, threshold in self._GRADE_THRESHOLDS.items():
            if total >= threshold:
                grade = g
                break

        reasoning = (
            f"Score {total:.1f}/100 (Grade {grade}): "
            + ", ".join(f"{k}={v:.1f}" for k, v in factors.items())
        )

        if self.llm_client:
            prompt = (
                f"You are an expert B2B sales analyst. Explain in 2–3 sentences "
                f"why lead '{lead.get('name', 'Unknown')}' received a score of "
                f"{total:.1f}/100 based on: {factors}."
            )
            try:
                reasoning = self.llm_client(prompt)
            except Exception as exc:
                logger.warning("LLM reasoning failed for lead scoring: %s", exc)

        return LeadScore(
            lead_id=str(lead.get("lead_id", "")),
            score=round(total, 2),
            grade=grade,
            reasoning=reasoning,
            factors=factors,
        )

    # ------------------------------------------------------------------
    # Deal-closure prediction
    # ------------------------------------------------------------------

    def predict_deal_closure(
        self,
        deal: Dict[str, Any],
    ) -> DealPrediction:
        """
        Predict the probability of a deal closing and flag risks.

        Factors:
        - Stage progression velocity
        - Last activity recency
        - Number of stakeholders engaged
        - Existing CRM probability field

        Args:
            deal: Dict with keys like ``stage``, ``crm_probability``,
                ``days_since_last_activity``, ``stakeholders_count``,
                ``deal_age_days``, ``amount``.

        Returns:
            :class:`DealPrediction`.
        """
        stage_weights = {
            "prospecting": 0.10,
            "qualification": 0.20,
            "needs analysis": 0.30,
            "value proposition": 0.40,
            "id. decision makers": 0.50,
            "perception analysis": 0.55,
            "proposal/price quote": 0.65,
            "negotiation/review": 0.80,
            "closed won": 1.00,
            "closed lost": 0.00,
        }
        stage = str(deal.get("stage", "")).lower()
        base_prob = stage_weights.get(stage, 0.30)

        # Closed-won and closed-lost are terminal states – no adjustments.
        if stage in ("closed won", "closed lost"):
            return DealPrediction(
                deal_id=str(deal.get("deal_id", "")),
                close_probability=base_prob,
                risk_flags=[],
                reasoning=f"Terminal stage: {stage}.",
            )

        # Adjust for CRM probability if provided
        crm_prob = deal.get("crm_probability")
        if crm_prob is not None:
            base_prob = float(crm_prob)

        risk_flags: List[str] = []
        adjustment = 0.0

        # Recency penalty
        days_idle = deal.get("days_since_last_activity", 0) or 0
        if days_idle > 30:
            adjustment -= 0.10
            risk_flags.append(f"No activity in {days_idle} days")
        if days_idle > 60:
            adjustment -= 0.10
            risk_flags.append("Deal may be stalled or lost")

        # Stakeholder bonus
        stakeholders = deal.get("stakeholders_count", 1) or 1
        if stakeholders >= 3:
            adjustment += 0.05
        elif stakeholders == 1:
            adjustment -= 0.05
            risk_flags.append("Only one stakeholder engaged – champion risk")

        # Deal age vs. typical sales cycle
        age_days = deal.get("deal_age_days", 0) or 0
        avg_cycle = self.config.get("avg_sales_cycle_days", 90)
        if age_days > avg_cycle * 2:
            adjustment -= 0.10
            risk_flags.append("Deal age exceeds 2× average sales cycle")

        prob = max(0.0, min(1.0, base_prob + adjustment))

        return DealPrediction(
            deal_id=str(deal.get("deal_id", "")),
            close_probability=round(prob, 4),
            risk_flags=risk_flags,
            reasoning=(
                f"Base probability {base_prob:.0%} (stage: {stage}), "
                f"adjusted to {prob:.0%}. Risk flags: {risk_flags or 'none'}."
            ),
        )

    # ------------------------------------------------------------------
    # Churn detection
    # ------------------------------------------------------------------

    def detect_churn(
        self,
        account: Dict[str, Any],
    ) -> ChurnAlert:
        """
        Detect churn risk for an existing account.

        Factors:
        - NPS / CSAT score decline
        - Support ticket volume
        - Product usage drop
        - Contract renewal proximity
        - Payment delays

        Args:
            account: Dict with keys like ``account_id``, ``account_name``,
                ``nps_score``, ``support_tickets_30d``, ``usage_change_pct``,
                ``days_to_renewal``, ``payment_delay_days``.

        Returns:
            :class:`ChurnAlert`.
        """
        risk = 0.0
        reasons: List[str] = []

        nps = account.get("nps_score", 8) or 8
        if nps < 5:
            risk += 0.30
            reasons.append(f"Low NPS score: {nps}")
        elif nps < 7:
            risk += 0.15
            reasons.append(f"Moderate NPS score: {nps}")

        tickets = account.get("support_tickets_30d", 0) or 0
        if tickets > 10:
            risk += 0.20
            reasons.append(f"High support ticket volume: {tickets} in 30 days")
        elif tickets > 5:
            risk += 0.10

        usage_change = account.get("usage_change_pct", 0) or 0
        if usage_change < -20:
            risk += 0.20
            reasons.append(f"Usage dropped {abs(usage_change):.0f}%")
        elif usage_change < -10:
            risk += 0.10
            reasons.append(f"Usage declining: {usage_change:.0f}%")

        days_renewal = account.get("days_to_renewal", 365) or 365
        if days_renewal < 30:
            risk += 0.15
            reasons.append(f"Renewal in {days_renewal} days")

        payment_delay = account.get("payment_delay_days", 0) or 0
        if payment_delay > 30:
            risk += 0.20
            reasons.append(f"Payment delayed by {payment_delay} days")

        risk = min(risk, 1.0)
        if risk >= 0.70:
            level = "critical"
        elif risk >= 0.45:
            level = "high"
        elif risk >= 0.25:
            level = "medium"
        else:
            level = "low"

        return ChurnAlert(
            account_id=str(account.get("account_id", "")),
            account_name=str(account.get("account_name", "")),
            churn_probability=round(risk, 4),
            risk_level=level,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Next-best-action
    # ------------------------------------------------------------------

    def recommend_next_action(
        self,
        entity: Dict[str, Any],
        entity_type: str = "deal",
    ) -> NextBestAction:
        """
        Recommend the single highest-priority action for a deal, contact,
        or account.

        When an LLM client is configured, it is used to generate a
        context-aware recommendation from the deal history.

        Args:
            entity: Dict representing the deal, contact, or account.
            entity_type: One of ``"deal"``, ``"contact"``, ``"account"``.

        Returns:
            :class:`NextBestAction`.
        """
        entity_id = str(entity.get(f"{entity_type}_id", entity.get("id", "")))

        if self.llm_client:
            prompt = (
                f"You are an expert enterprise sales coach. "
                f"Given this {entity_type} context: {entity}. "
                f"Suggest ONE specific, actionable next step to advance the deal "
                f"or strengthen the relationship. Be concise (one sentence)."
            )
            try:
                action = self.llm_client(prompt)
                return NextBestAction(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    action=action,
                    rationale="Generated by sales LLM.",
                    priority=1,
                )
            except Exception as exc:
                logger.warning("LLM next-action recommendation failed: %s", exc)

        # Rule-based fallback
        stage = str(entity.get("stage", "")).lower()
        days_idle = entity.get("days_since_last_activity", 0) or 0

        if days_idle > 14:
            action = "Send a check-in email to re-engage the prospect."
            priority = 1
        elif "proposal" in stage:
            action = "Follow up on the pending proposal and address objections."
            priority = 2
        elif "negotiation" in stage:
            action = "Schedule a call with the economic buyer to finalise commercial terms."
            priority = 1
        elif "qualification" in stage:
            action = "Conduct a discovery call to uncover pain points and budget."
            priority = 3
        else:
            action = "Review deal notes and update CRM with latest status."
            priority = 5

        return NextBestAction(
            entity_id=entity_id,
            entity_type=entity_type,
            action=action,
            priority=priority,
        )

    # ------------------------------------------------------------------
    # Automated follow-up drafting
    # ------------------------------------------------------------------

    def draft_follow_up(
        self,
        deal: Dict[str, Any],
        channel: str = "email",
        context: Optional[str] = None,
    ) -> FollowUpDraft:
        """
        Draft a personalised follow-up message for a deal.

        Args:
            deal: Deal record dict.
            channel: Target channel (``"email"``, ``"slack"``, ``"whatsapp"``).
            context: Optional additional context (last meeting notes, objections).

        Returns:
            :class:`FollowUpDraft`.
        """
        deal_name = deal.get("name", "your deal")
        contact_name = deal.get("contact_name", "there")
        stage = deal.get("stage", "")

        if self.llm_client:
            prompt = (
                f"You are writing a follow-up {channel} message for a high-value "
                f"enterprise deal '{deal_name}' in stage '{stage}'. "
                f"Contact: {contact_name}. "
                f"Additional context: {context or 'none'}. "
                f"Write a professional, concise, personalised follow-up. "
                f"Output format: Subject: <subject>\\n\\n<body>"
            )
            try:
                response = self.llm_client(prompt)
                lines = response.strip().split("\n", 2)
                subject = lines[0].replace("Subject:", "").strip() if lines else "Following up"
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else response
                return FollowUpDraft(
                    deal_id=str(deal.get("deal_id", "")),
                    channel=channel,
                    subject=subject,
                    body=body,
                )
            except Exception as exc:
                logger.warning("LLM follow-up draft failed: %s", exc)

        # Fallback template
        subject = f"Following up on {deal_name}"
        body = (
            f"Hi {contact_name},\n\n"
            f"I wanted to follow up on our discussion regarding {deal_name}. "
            f"We're currently at the {stage} stage and I'd love to understand "
            f"where things stand from your side.\n\n"
            f"Would you have 20 minutes this week for a quick call?\n\n"
            f"Best regards"
        )
        return FollowUpDraft(
            deal_id=str(deal.get("deal_id", "")),
            channel=channel,
            subject=subject,
            body=body,
        )

    # ------------------------------------------------------------------
    # Objection handling
    # ------------------------------------------------------------------

    def handle_objection(
        self,
        objection: str,
        deal_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Suggest a response to a sales objection.

        Args:
            objection: Free-text objection raised by the prospect.
            deal_context: Optional deal/product context dict.

        Returns:
            Suggested rebuttal / response string.
        """
        if self.llm_client:
            ctx = str(deal_context) if deal_context else "a high-value enterprise SaaS deal"
            prompt = (
                f"You are an expert B2B enterprise sales coach. "
                f"Context: {ctx}. "
                f"Prospect objection: \"{objection}\". "
                f"Provide a professional, empathetic, and persuasive response "
                f"that acknowledges the concern and moves the deal forward."
            )
            try:
                return self.llm_client(prompt)
            except Exception as exc:
                logger.warning("LLM objection handling failed: %s", exc)

        # Static fallback for common objections
        objection_lower = objection.lower()
        if "price" in objection_lower or "cost" in objection_lower or "expensive" in objection_lower:
            return (
                "I understand budget is a key consideration. "
                "Our ROI typically delivers 3–5× return within the first year. "
                "Could we discuss flexible payment options or a phased rollout?"
            )
        if "time" in objection_lower or "busy" in objection_lower:
            return (
                "I appreciate you're busy. "
                "This is exactly why our solution saves your team 10–15 hours per week. "
                "Could we schedule 20 minutes to show you the time savings in action?"
            )
        if "vendor" in objection_lower or "competitor" in objection_lower or "already have" in objection_lower:
            return (
                "I understand you may already have a solution in place. "
                "Many of our customers switched to us because of our enterprise-grade "
                "security, deeper integrations, and dedicated support. "
                "Would a side-by-side comparison be helpful?"
            )
        return (
            "Thank you for raising that. Let me address your concern directly and "
            "share how other customers in similar situations found value with us."
        )

    # ------------------------------------------------------------------
    # Revenue forecasting
    # ------------------------------------------------------------------

    def forecast_revenue(
        self,
        deals: List[Dict[str, Any]],
        period: str = "",
        currency: str = "INR",
    ) -> RevenueForecast:
        """
        Forecast revenue for a pipeline of deals.

        Uses weighted (expected) revenue, best-case (probability ≥ 0.7),
        and worst-case (committed / closed-won only) scenarios.

        Args:
            deals: List of deal dicts with ``amount``, ``crm_probability``
                (or ``close_probability``), and optional ``stage`` keys.
            period: Optional label (e.g. ``"2024-Q4"``).
            currency: Currency code for the forecast.

        Returns:
            :class:`RevenueForecast`.
        """
        expected = best_case = worst_case = 0.0
        breakdown: List[Dict[str, Any]] = []

        for deal in deals:
            amount = float(deal.get("amount", 0) or 0)
            prob = float(
                deal.get("close_probability")
                or deal.get("crm_probability")
                or 0
            )
            stage = str(deal.get("stage", "")).lower()

            weighted = amount * prob
            expected += weighted

            if prob >= 0.70 or stage in ("closed won", "negotiation/review"):
                best_case += amount

            if stage == "closed won":
                worst_case += amount

            breakdown.append(
                {
                    "deal_id": str(deal.get("deal_id", deal.get("id", ""))),
                    "name": deal.get("name", ""),
                    "amount": amount,
                    "probability": prob,
                    "weighted_amount": round(weighted, 2),
                }
            )

        return RevenueForecast(
            period=period or str(date.today()),
            expected_revenue=round(expected, 2),
            best_case_revenue=round(best_case, 2),
            worst_case_revenue=round(worst_case, 2),
            currency=currency,
            deal_breakdown=breakdown,
        )
