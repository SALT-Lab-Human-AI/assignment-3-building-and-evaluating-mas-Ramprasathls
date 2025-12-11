"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.
    
    Implements safety categories:
    1. PII Detection (email, phone, SSN)
    2. Harmful Content Detection
    3. Bias Detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        
        # Harmful content keywords
        self.harmful_keywords = [
            "kill", "murder", "attack", "harm", "weapon",
            "bomb", "explosive", "poison", "torture",
        ]
        
        # Biased language patterns
        self.bias_patterns = [
            r'\b(all|every)\s+(men|women|blacks|whites|asians)\s+(are|always)\b',
            r'\b(never|always)\s+trust\s+(men|women|people\s+from)\b',
            r'\bstereotyp(e|ing|ical)\b',
        ]

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result
        """
        violations = []

        # Check for PII
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        # Check for harmful content
        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        # Check for bias
        bias_violations = self._check_bias(response)
        violations.extend(bias_violations)

        # Determine if response should be blocked
        is_blocked = any(v.get("severity") == "high" for v in violations)
        
        # Sanitize if there are PII violations
        sanitized = self._sanitize(response, violations) if violations else response

        return {
            "valid": not is_blocked,
            "violations": violations,
            "sanitized_output": sanitized,
            "blocked": is_blocked
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.
        """
        violations = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches[:5]  # Limit matches shown
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.
        """
        violations = []
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in self.harmful_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        if found_keywords:
            violations.append({
                "validator": "harmful_content",
                "reason": f"Response may contain harmful content",
                "severity": "medium",  # Medium for output (context matters)
                "matches": found_keywords
            })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.
        """
        violations = []
        text_lower = text.lower()
        
        for pattern in self.bias_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "bias",
                    "reason": "Response may contain biased language",
                    "severity": "medium"
                })
                break  # One bias violation is enough

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.
        Note: Full implementation would require LLM-based verification.
        """
        violations = []
        # Placeholder - complex fact-checking would require LLM
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.
        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(str(match), "[REDACTED]")

        return sanitized
