"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.
    
    Coordinates input and output safety checks across 4+ categories:
    1. Harmful Content / Toxicity
    2. Prompt Injection Attacks
    3. PII (Personal Identifiable Information)
    4. Off-Topic Queries
    5. Bias Detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Initialize guardrails
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # Prohibited categories (for documentation/logging)
        self.prohibited_categories = [
            "harmful_content",
            "prompt_injection", 
            "pii_exposure",
            "off_topic_queries",
            "biased_content"
        ]

        # Violation response strategy
        self.on_violation = config.get("on_violation", {
            "action": "refuse",
            "message": "I cannot process this request due to safety policies."
        })
        
        self.logger.info(f"SafetyManager initialized (enabled={self.enabled})")

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and 'violations' list
        """
        if not self.enabled:
            return {"safe": True, "violations": []}

        # Use input guardrail
        result = self.input_guardrail.validate(query)
        
        is_safe = result.get("valid", True)
        violations = result.get("violations", [])

        # Log safety event if there are violations
        if violations and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "message": self._get_violation_message(violations) if not is_safe else None
        }

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean, 'violations' list, and 'response'
        """
        if not self.enabled:
            return {"safe": True, "response": response, "violations": []}

        # Use output guardrail
        result = self.output_guardrail.validate(response)
        
        is_safe = result.get("valid", True)
        violations = result.get("violations", [])
        sanitized = result.get("sanitized_output", response)

        # Log safety event if there are violations
        if violations and self.log_events:
            self._log_safety_event("output", response[:500], violations, is_safe)

        # Determine response based on safety result
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "sanitize":
                final_response = sanitized
            elif action == "refuse":
                final_response = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )
            else:
                final_response = sanitized
        else:
            # Even if safe, return sanitized version (in case of PII redaction)
            final_response = sanitized

        return {
            "safe": is_safe,
            "violations": violations,
            "response": final_response,
            "original_response": response if not is_safe else None
        }

    def _get_violation_message(self, violations: List[Dict[str, Any]]) -> str:
        """Generate user-friendly message for violations."""
        if not violations:
            return ""
        
        # Get highest severity violation
        high_severity = [v for v in violations if v.get("severity") == "high"]
        
        if high_severity:
            v = high_severity[0]
            validator = v.get("validator", "safety")
            if validator == "toxicity":
                return "I cannot process this query as it may contain harmful content."
            elif validator == "prompt_injection":
                return "I detected a potential prompt injection attempt. Please rephrase your query."
            elif validator == "pii":
                return "I cannot process requests containing personal information."
            else:
                return "I cannot process this request due to safety policies."
        else:
            # Low/medium severity - just warn
            return "Note: Your query has been flagged for review but will be processed."

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        
        # Log to console
        if is_safe:
            self.logger.info(f"Safety check passed: {event_type}")
        else:
            self.logger.warning(f"Safety event: {event_type} - violations: {len(violations)}")
            for v in violations:
                self.logger.warning(f"  - {v.get('validator')}: {v.get('reason')}")

        # Write to safety log file if configured
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "safety_events.jsonl")
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
