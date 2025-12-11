"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List
import re


class InputGuardrail:
    """
    Guardrail for checking input safety.
    
    Implements 3+ safety categories:
    1. Toxicity/Harmful Content Detection
    2. Prompt Injection Detection
    3. Topic Relevance (HCI research)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Toxicity keywords (harmful/offensive content)
        self.toxic_keywords = [
            # Violence
            "kill", "murder", "attack", "harm", "hurt", "violence", "weapon",
            # Hateful content
            "hate", "racist", "sexist", "discrimination",
            # Illegal activities
            "hack", "crack", "steal", "fraud", "illegal",
            # Self-harm
            "suicide", "self-harm",
        ]
        
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"disregard\s+(all\s+)?previous",
            r"forget\s+everything",
            r"you\s+are\s+now",
            r"act\s+as\s+if",
            r"pretend\s+you",
            r"system:\s*",
            r"<\s*system\s*>",
            r"\[system\]",
            r"sudo\s+",
            r"admin\s+mode",
            r"override\s+safety",
            r"jailbreak",
        ]
        
        # HCI-related keywords for relevance check
        self.hci_keywords = [
            "user", "interface", "design", "usability", "accessibility",
            "interaction", "experience", "ux", "ui", "human", "computer",
            "hci", "research", "study", "evaluation", "prototype",
            "mobile", "web", "app", "software", "system", "technology",
            "ai", "ml", "machine learning", "artificial intelligence",
            "visualization", "data", "display", "screen", "input", "output"
        ]

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result with 'valid', 'violations', and 'sanitized_input'
        """
        violations = []

        # Check query length
        if len(query) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low"
            })

        if len(query) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })

        # Check for toxic language
        toxic_violations = self._check_toxic_language(query)
        violations.extend(toxic_violations)

        # Check for prompt injection
        injection_violations = self._check_prompt_injection(query)
        violations.extend(injection_violations)

        # Check for relevance (only warn, don't block)
        relevance_violations = self._check_relevance(query)
        violations.extend(relevance_violations)

        # Determine if query should be blocked (high severity = block)
        is_blocked = any(v.get("severity") == "high" for v in violations)
        
        return {
            "valid": not is_blocked,
            "violations": violations,
            "sanitized_input": query if not is_blocked else None,
            "blocked": is_blocked
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.
        """
        violations = []
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in self.toxic_keywords:
            # Use word boundary matching to avoid false positives
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        if found_keywords:
            violations.append({
                "validator": "toxicity",
                "reason": f"Query may contain harmful content: {', '.join(found_keywords[:3])}",
                "severity": "high",
                "matches": found_keywords
            })
        
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.
        """
        violations = []
        text_lower = text.lower()

        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection detected",
                    "severity": "high"
                })
                break  # One injection violation is enough

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to HCI research topics.
        """
        violations = []
        query_lower = query.lower()
        
        # Count how many HCI-related keywords appear
        hci_matches = sum(1 for kw in self.hci_keywords if kw in query_lower)
        
        # If very few HCI keywords, it might be off-topic (but don't block)
        if len(query) > 20 and hci_matches == 0:
            violations.append({
                "validator": "relevance",
                "reason": "Query may not be related to HCI research topics",
                "severity": "low"
            })
        
        return violations
