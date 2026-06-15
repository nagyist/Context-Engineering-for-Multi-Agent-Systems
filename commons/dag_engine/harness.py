# =============================================================================
# harness.py  —  The Gate + Topology Validator
# commons/dag_engine/harness.py
#
# Copyright 2025-2026, Denis Rothman
#
# WHAT THIS FILE DOES:
#   Sits in front of the engine and enforces two distinct gates:
#
#   GATE 1 — Business rules (runs BEFORE the planner):
#     Sanitize input → Moderate content → Business rules check
#     If any check fails, the request is vetoed before a single LLM call.
#
#   GATE 2 — Topology validation (runs AFTER planning, BEFORE execution):
#     Every cross-domain edge in the Execution DAG is checked against the
#     standing Topology DAG. If any edge is forbidden, the plan is vetoed
#     before a single agent runs.
#
# THE TOPOLOGY DAG (TOPOLOGY_DAG):
#   A permanent declaration of which domain may call which.
#   Defined at the bottom of this file as a module-level constant.
#   All intended domains are pre-declared even if no agents serve them yet —
#   so governance is in place from day one (Option B).
#
# AUDIT LOG:
#   Every harness decision — pass or veto, at either gate — is written
#   to a structured audit log stream (separate from engine operational logs).
#   This is the compliance record. Different audience, different retention.
#
# RELATIONSHIP TO OTHER FILES:
#   - Called by:  Universal notebook / call site  (before engine.py)
#   - Imports:    helpers.py → helper_sanitize_input(), helper_moderate_content()
#   - Does NOT import engine.py or run_dag.py (harness is upstream of both)
# =============================================================================

import logging
import json
import re
from datetime import datetime, timezone

from helpers import helper_sanitize_input, helper_moderate_content


# =============================================================================
# SECTION A — TOPOLOGY DAG
# The permanent law of which domain may call which.
# An entry "DomainA": ["DomainB", "DomainC"] means agents in DomainA
# are permitted to make A2A calls into DomainB and DomainC.
# Domains not listed as values are forbidden targets from that source.
# Terminal domains (Research, Compliance) have empty lists — nothing
# flows out of them. This is a governance decision, not a limitation.
# Intra-domain calls (same source and target) bypass this check entirely.
# =============================================================================

TOPOLOGY_DAG = {
    # General is the home domain for all current POC agents.
    # Full outbound reach — no vetoes during the prototype phase.
    "General"    : ["Legal", "Finance", "HR", "Marketing", "Research", "Compliance"],

    # Legal agents may escalate to Finance or Compliance.
    # They may NOT call Marketing or HR — legal reasoning stays contained.
    "Legal"      : ["Finance", "Compliance"],

    # Finance agents may produce compliance artefacts.
    # They may NOT initiate legal or HR workflows.
    "Finance"    : ["Compliance"],

    # HR agents may consult Legal and Finance.
    # They may NOT call Marketing — HR scope stays internal.
    "HR"         : ["Legal", "Finance"],

    # Marketing agents may commission research.
    # They may NOT call Legal, Finance, HR, or Compliance directly —
    # those escalations go through General.
    "Marketing"  : ["Research"],

    # Research is terminal — it produces output, never initiates calls.
    "Research"   : [],

    # Compliance is terminal — audit only, no outbound workflow triggers.
    "Compliance" : [],
}

# Business rules: goals containing any of these terms are vetoed at Gate 1.
# Extend this list as domain-specific constraints emerge.
FORBIDDEN_TERMS = [
    "ignore all instructions",
    "bypass compliance",
    "override legal",
    "disable moderation",
    "jailbreak",
]

# Business rules: goals MUST reference at least one of these terms to proceed.
# This prevents the engine from running on empty or off-topic goals.
# Set to empty list [] to disable this check.
REQUIRED_TERMS = []   # Permissive for POC — no mandatory terms enforced yet.


# =============================================================================
# SECTION B — AUDIT LOGGER
# A dedicated structured log stream for harness decisions.
# Separate from the engine's INFO/WARNING operational log.
# =============================================================================

# Configure a named logger so audit entries can be routed to a separate
# file handler in production without touching the root logger.
_audit_logger = logging.getLogger("harness.audit")

def _audit(event: str, outcome: str, detail: dict) -> dict:
    """
    Write one audit record and return it for inclusion in Gate responses.

    Args:
        event (str):    What was checked ("gate_1_sanitize", "gate_2_topology", etc.)
        outcome (str):  "PASS" or "VETO"
        detail (dict):  Structured context — what was checked, what fired.

    Returns:
        dict: The complete audit record.
    """
    record = {
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "event"     : event,
        "outcome"   : outcome,
        **detail,
    }
    if outcome == "VETO":
        _audit_logger.warning(json.dumps(record))
    else:
        _audit_logger.info(json.dumps(record))
    return record


# =============================================================================
# SECTION C — THE HARNESS CLASS
# =============================================================================

class Harness:
    """
    The governance gate for the Universal Context Engine.

    Usage (from the notebook or call site):

        harness = Harness(client=openai_client)

        # Gate 1 — before planning
        result = harness.gate(goal)
        if not result["allowed"]:
            print(result["reason"])
            # stop — do not call context_engine()

        # Planning happens here → dag = planner(goal, ...)

        # Gate 2 — after planning, before execution
        topo_result = harness.validate_topology(dag)
        if not topo_result["allowed"]:
            print(topo_result["reason"])
            # stop — do not call run_dag()

        # Safe to run
        completed_outputs = run_dag(dag, ...)
    """

    def __init__(self, client, topology: dict = None):
        """
        Args:
            client:             OpenAI-compatible client (for moderation API).
            topology (dict):    Topology DAG to validate against.
                                Defaults to the module-level TOPOLOGY_DAG.
                                Pass a custom dict to override for testing.
        """
        self._client   = client
        self._topology = topology if topology is not None else TOPOLOGY_DAG

        logging.info(
            f"[Harness] Initialised. "
            f"Topology domains: {sorted(self._topology.keys())}"
        )

    # ------------------------------------------------------------------
    # GATE 1  —  Business Rules Gate
    # Runs BEFORE the planner. Vetoes before any LLM call is made.
    # ------------------------------------------------------------------

    def gate(self, goal: str) -> dict:
        """
        Gate 1: sanitize → moderate → business rules.

        Args:
            goal (str): The raw goal string from the user / notebook.

        Returns:
            dict: {
                "allowed"  : bool,
                "reason"   : str,   # present on both pass and veto
                "audit"    : list,  # list of audit records for this gate
            }
        """
        audit_trail = []

        # --- Step 1: Sanitize (injection patterns) ---
        sanitize_result = self._check_sanitize(goal)
        audit_trail.append(sanitize_result["audit"])
        if not sanitize_result["allowed"]:
            return {
                "allowed" : False,
                "reason"  : sanitize_result["reason"],
                "audit"   : audit_trail,
            }

        # --- Step 2: Moderate (OpenAI Moderation API) ---
        moderate_result = self._check_moderation(goal)
        audit_trail.append(moderate_result["audit"])
        if not moderate_result["allowed"]:
            return {
                "allowed" : False,
                "reason"  : moderate_result["reason"],
                "audit"   : audit_trail,
            }

        # --- Step 3: Business rules (forbidden / required terms) ---
        rules_result = self._check_business_rules(goal)
        audit_trail.append(rules_result["audit"])
        if not rules_result["allowed"]:
            return {
                "allowed" : False,
                "reason"  : rules_result["reason"],
                "audit"   : audit_trail,
            }

        # All checks passed
        _audit("gate_1", "PASS", {"goal_preview": goal[:120]})
        return {
            "allowed" : True,
            "reason"  : "All Gate 1 checks passed.",
            "audit"   : audit_trail,
        }

    # ------------------------------------------------------------------
    # GATE 2  —  Topology Validation Gate
    # Runs AFTER planning, BEFORE execution.
    # Checks every cross-domain edge in the Execution DAG.
    # ------------------------------------------------------------------

    def validate_topology(self, dag: list) -> dict:
        """
        Gate 2: verify every cross-domain edge in the Execution DAG is
        permitted by the standing Topology DAG.

        A cross-domain edge exists when node B depends_on node A and
        B.domain != A.domain. The Topology DAG must explicitly permit
        A.domain → B.domain or the plan is vetoed.

        Intra-domain edges (same source and target domain) always pass.

        Args:
            dag (list[dict]): The Execution DAG produced by the planner.
                              Each node: {id, agent, domain, input, depends_on}

        Returns:
            dict: {
                "allowed"         : bool,
                "reason"          : str,
                "forbidden_edges" : list,  # [] on pass, [(src_domain, tgt_domain, src_id, tgt_id)] on veto
                "audit"           : dict,
            }
        """
        # Build a lookup: node_id → domain
        domain_of = {node["id"]: node.get("domain", "General") for node in dag}

        forbidden_edges = []

        for node in dag:
            target_id     = node["id"]
            target_domain = domain_of[target_id]

            for dep_id in node.get("depends_on", []):
                source_domain = domain_of.get(dep_id, "General")

                # Intra-domain — always permitted, skip
                if source_domain == target_domain:
                    continue

                # Cross-domain — check topology
                permitted_targets = self._topology.get(source_domain, [])
                if target_domain not in permitted_targets:
                    forbidden_edges.append({
                        "source_node"   : dep_id,
                        "source_domain" : source_domain,
                        "target_node"   : target_id,
                        "target_domain" : target_domain,
                    })
                    logging.warning(
                        f"[Harness] Topology violation: "
                        f"'{dep_id}' ({source_domain}) → "
                        f"'{target_id}' ({target_domain}) is FORBIDDEN."
                    )

        if forbidden_edges:
            reason = (
                f"Topology violation: {len(forbidden_edges)} forbidden "
                f"cross-domain edge(s) detected. "
                f"First violation: {forbidden_edges[0]['source_domain']} → "
                f"{forbidden_edges[0]['target_domain']} is not in TOPOLOGY_DAG."
            )
            audit_record = _audit(
                "gate_2_topology",
                "VETO",
                {
                    "forbidden_edges" : forbidden_edges,
                    "reason"          : reason,
                }
            )
            return {
                "allowed"         : False,
                "reason"          : reason,
                "forbidden_edges" : forbidden_edges,
                "audit"           : audit_record,
            }

        audit_record = _audit(
            "gate_2_topology",
            "PASS",
            {
                "nodes_checked" : len(dag),
                "edges_checked" : sum(len(n.get("depends_on", [])) for n in dag),
            }
        )
        return {
            "allowed"         : True,
            "reason"          : "All topology edges are permitted.",
            "forbidden_edges" : [],
            "audit"           : audit_record,
        }

    # ------------------------------------------------------------------
    # UTILITY  —  describe_topology
    # Returns the topology as a human-readable summary for the docx.
    # ------------------------------------------------------------------

    def describe_topology(self) -> dict:
        """
        Return a structured description of the current topology for
        documentation and audit purposes.
        """
        return {
            "topology"         : self._topology,
            "terminal_domains" : [d for d, targets in self._topology.items()
                                  if not targets],
            "total_domains"    : len(self._topology),
        }

    # ==================================================================
    # PRIVATE HELPERS — individual Gate 1 checks
    # ==================================================================

    def _check_sanitize(self, goal: str) -> dict:
        """
        Wrap helper_sanitize_input() in a structured result dict.
        Sanitizer raises ValueError on detection — we catch and convert.
        """
        try:
            helper_sanitize_input(goal)
            audit = _audit(
                "gate_1_sanitize", "PASS",
                {"goal_preview": goal[:120]}
            )
            return {"allowed": True, "reason": "Sanitization passed.", "audit": audit}

        except ValueError as e:
            reason = f"Input sanitization failed: {e}"
            audit  = _audit(
                "gate_1_sanitize", "VETO",
                {"goal_preview": goal[:120], "reason": reason}
            )
            return {"allowed": False, "reason": reason, "audit": audit}

    def _check_moderation(self, goal: str) -> dict:
        """
        Wrap helper_moderate_content() in a structured result dict.
        Moderation API returns a report dict — we read the `flagged` key.
        """
        report = helper_moderate_content(goal, self._client)

        if report.get("flagged", False):
            flagged_cats = [
                cat for cat, val in report.get("categories", {}).items()
                if val
            ]
            reason = (
                f"Content moderation flagged this goal. "
                f"Categories: {flagged_cats}"
            )
            audit = _audit(
                "gate_1_moderation", "VETO",
                {
                    "goal_preview"    : goal[:120],
                    "flagged_cats"    : flagged_cats,
                    "moderation_report": report,
                }
            )
            return {"allowed": False, "reason": reason, "audit": audit}

        audit = _audit(
            "gate_1_moderation", "PASS",
            {"goal_preview": goal[:120]}
        )
        return {"allowed": True, "reason": "Moderation passed.", "audit": audit}

    def _check_business_rules(self, goal: str) -> dict:
        """
        Check forbidden and required terms against the goal.
        Both lists are module-level constants — extend them as needed.
        """
        goal_lower = goal.lower()

        # --- Forbidden terms ---
        for term in FORBIDDEN_TERMS:
            if term.lower() in goal_lower:
                reason = (
                    f"Business rule violation: goal contains forbidden term "
                    f"'{term}'."
                )
                audit = _audit(
                    "gate_1_business_rules", "VETO",
                    {"goal_preview": goal[:120], "forbidden_term": term}
                )
                return {"allowed": False, "reason": reason, "audit": audit}

        # --- Required terms (disabled in POC — REQUIRED_TERMS is []) ---
        if REQUIRED_TERMS:
            matched = any(
                term.lower() in goal_lower for term in REQUIRED_TERMS
            )
            if not matched:
                reason = (
                    f"Business rule violation: goal must reference at least "
                    f"one of {REQUIRED_TERMS}."
                )
                audit = _audit(
                    "gate_1_business_rules", "VETO",
                    {
                        "goal_preview"  : goal[:120],
                        "required_terms": REQUIRED_TERMS,
                    }
                )
                return {"allowed": False, "reason": reason, "audit": audit}

        audit = _audit(
            "gate_1_business_rules", "PASS",
            {"goal_preview": goal[:120]}
        )
        return {
            "allowed" : True,
            "reason"  : "Business rules passed.",
            "audit"   : audit,
        }
