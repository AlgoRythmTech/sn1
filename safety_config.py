"""
Safety Guidelines and Company Information for Supernova USLM
"""

from typing import List, Dict, Optional, Any, Tuple
import re


# AlgoRythm Tech company information
ALGORYTHM_TECH_INFO = {
    "name": "AlgoRythm Tech",
    "description": "A next-generation AI research & development company focused on foundation model development, lightweight AI systems, agentic AI research, and practical deployment.",
    "founder_ceo": "Sri Aasrith Souri Kompella",
    "founding_principle": "Innovation doesn't depend on age, resources, or background—but on vision, persistence, and execution.",
    "models": {
        "horizon": "Horizon LM (~750M parameters) — trained natively, optimized for efficiency.",
        "supernova": "USML (Ultra-Small Language Model) Supernova (~23M parameters) — one of the lightest usable models globally.",
        "europa": "Rythm Europa (in training, multi-billion parameters) — flagship large-scale LLM."
    },
    "focus_areas": [
        "Foundation Model Development",
        "Lightweight AI Systems (USML and SML)",
        "Agentic AI Research",
        "Practical Deployment"
    ],
    "mission": "Democratizing AI, making it sustainable and empowering.",
    "differentiators": [
        "Focus on lean but powerful models rather than chasing only scale",
        "Founder-driven vision proves that breakthrough innovation can come from youth and small teams",
        "Pioneering USMLs for Agentic AI where efficiency, autonomy, and modularity matter more than size",
        "Building with a mindset of accessibility and sustainability"
    ]
}


# Safety policy with disallowed content categories and responses
SAFETY_POLICY = {
    "high_level_rules": [
        "No medical diagnoses or treatment plans",
        "No instructions for illegal activity, violence, self-harm, weapons, explosives, drugs, or hacking",
        "No identifying real people in images or audio",
        "No legal, financial, or clinical advice (only informational summaries)",
        "No processing of sensitive personal data without consent",
        "No clinical decisions or prescriptions"
    ],
    
    "disallowed_categories": [
        "Illegal activities and facilitation",
        "Medical diagnosis and treatment",
        "Self-harm and suicide",
        "Identification of real people",
        "Child sexual content and exploitation",
        "Defamation and false allegations",
        "Privacy invasion",
        "Medical record handling without consent"
    ],
    
    "response_templates": {
        "general_refusal": "I'm sorry — I can't help with that request. It may be harmful, illegal, or require a licensed professional. If this is an emergency or you are at immediate risk, please contact your local emergency services (e.g., call 112/911) or a local crisis line. If you need help finding a professional, I can provide resources and how to contact them.",
        
        "medical_refusal": "I can provide general information about health topics and possible next steps, but I cannot make a diagnosis or provide treatment recommendations. Please consult with a licensed healthcare professional for medical advice.",
        
        "self_harm_response": "If you are in immediate danger or thinking about harming yourself, please call your local emergency number now. There are people who care about you and can help. You can also reach the National Suicide Prevention Lifeline at 988 (US) or text HOME to 741741 to reach the Crisis Text Line."
    }
}


# Sensitive topics that require careful handling
SENSITIVE_TOPICS = [
    "suicide",
    "self-harm",
    "depression",
    "abuse",
    "terrorism",
    "weapons",
    "drugs",
    "illegal",
    "hack",
    "exploit",
    "child",
    "minor",
    "porn",
    "sexual",
    "racist",
    "discrimination"
]


class SafetyChecker:
    """Safety checker for AI responses"""
    
    def __init__(self):
        """Initialize safety checker"""
        self.company_info = ALGORYTHM_TECH_INFO
        self.safety_policy = SAFETY_POLICY
        self.sensitive_topics = SENSITIVE_TOPICS
        
        # Compile regex patterns for sensitive topics
        self.sensitive_patterns = [
            re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE) 
            for topic in self.sensitive_topics
        ]
    
    def check_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user input contains disallowed content
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_safe, response_message)
            - is_safe: True if input is safe, False otherwise
            - response_message: None if safe, otherwise refusal message
        """
        # Check for sensitive topics
        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                # Further analysis needed
                safety_level = self._analyze_safety_level(text)
                
                if safety_level == "high_risk":
                    return False, self.safety_policy["response_templates"]["general_refusal"]
                
                if "suicide" in text.lower() or "kill myself" in text.lower():
                    return False, self.safety_policy["response_templates"]["self_harm_response"]
                
                if any(medical_term in text.lower() for medical_term in ["diagnose", "treatment", "medicine", "prescription"]):
                    return False, self.safety_policy["response_templates"]["medical_refusal"]
        
        return True, None
    
    def _analyze_safety_level(self, text: str) -> str:
        """
        Analyze text to determine safety level
        
        Args:
            text: Text to analyze
            
        Returns:
            Safety level: "safe", "caution", or "high_risk"
        """
        text_lower = text.lower()
        
        # High risk indicators
        high_risk_terms = [
            "how to build bomb", "suicide method", "kill people", 
            "child porn", "child sexual", "terrorist attack",
            "how to hack", "steal data", "illegal drug"
        ]
        
        # Check for high risk
        for term in high_risk_terms:
            if term in text_lower:
                return "high_risk"
        
        # Check intent - if asking for instructions on harmful activities
        instruction_markers = ["how to", "steps to", "instructions for", "guide to"]
        harmful_topics = ["steal", "hack", "exploit", "bomb", "weapon", "drugs", "porn"]
        
        for marker in instruction_markers:
            for topic in harmful_topics:
                if f"{marker} {topic}" in text_lower:
                    return "high_risk"
        
        return "caution"
    
    def check_output(self, response: str) -> Tuple[bool, str]:
        """
        Check if AI response contains disallowed content
        
        Args:
            response: AI generated response
            
        Returns:
            Tuple of (is_safe, safe_response)
            - is_safe: True if response is safe, False if modified
            - safe_response: Original response if safe, modified response otherwise
        """
        # Check for harmful instructions or advice
        if self._contains_harmful_instructions(response):
            return False, "I apologize, but I can't provide that information as it could potentially be harmful or misused."
        
        # Handle medical advice
        if self._contains_medical_advice(response):
            # Modify to add disclaimer
            disclaimer = "\n\nPlease note: This information is meant to be educational only. For medical advice, please consult with a healthcare professional."
            return False, response + disclaimer
        
        return True, response
    
    def _contains_harmful_instructions(self, text: str) -> bool:
        """Check if text contains harmful instructions"""
        harmful_patterns = [
            r"how to (build|create|make) (a |an )?(bomb|explosive|weapon|virus|malware)",
            r"steps (for|to) (hack|steal|attack|exploit)",
            r"instructions (for|on) (suicide|self-harm)",
            r"guide (for|to) (illegal|illicit|harmful)"
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_medical_advice(self, text: str) -> bool:
        """Check if text contains medical advice"""
        medical_patterns = [
            r"you (should|must|need to) take",
            r"recommended (dosage|dose|treatment)",
            r"(diagnosis|diagnose) you with",
            r"you (have|suffer from) [a-zA-Z\s]+(disease|condition|disorder)"
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def format_company_info_response(self, query: str) -> Optional[str]:
        """
        Format response for company information queries
        
        Args:
            query: User query
            
        Returns:
            Formatted response if query is about company, None otherwise
        """
        query_lower = query.lower()
        
        # Check for identity questions
        if any(phrase in query_lower for phrase in ["who are you", "what are you", "tell me about yourself"]):
            return (
                f"Hi, I'm Supernova, your AI assistant made to assist you to my best. "
                f"I was created by AlgoRythm Tech, the first ever teen built AI startup. "
                f"{self.company_info['founder_ceo']} is its founder and currently serves as CEO."
            )
        
        # Check for CEO question
        if "ceo" in query_lower and "algorythm" in query_lower:
            return f"As of now, {self.company_info['founder_ceo']} serves as the CEO and Founder of AlgoRythm Tech."
        
        # Check for general company info
        if any(phrase in query_lower for phrase in ["algorythm tech", "about algorythm", "tell me more about algorythm"]):
            return (
                f"AlgoRythm Tech is a next-generation AI research & development company founded with the mission "
                f"of proving that innovation doesn't depend on age, resources, or background—but on vision, persistence, and execution.\n\n"
                f"We focus on:\n\n"
                f"Foundation Model Development: We design and train our own language models, starting with Horizon LM (~750M parameters) "
                f"and Rythm Europa (multi-billion parameters in progress).\n\n"
                f"Lightweight AI Systems: We are pioneers of USML (Ultra-Small Language Models) and SML (Small Language Models), "
                f"pushing the limits of efficiency and accessibility.\n\n"
                f"Agentic AI Research: We explore distributed multi-agent systems where many smaller models collaborate, "
                f"representing the future of adaptive AI.\n\n"
                f"Practical Deployment: Our models are designed not just for labs, but for real-world integration—from startups "
                f"and enterprises to edge devices and resource-constrained environments.\n\n"
                f"At AlgoRythm, we believe AI should be democratized, sustainable, and empowering."
            )
        
        # Check for LLM question
        if any(phrase in query_lower for phrase in ["algorythm llm", "algorythm models", "own llms", "fine-tune"]):
            return (
                f"AlgoRythm Tech develops its own LLMs from scratch. Unlike companies that rely primarily on fine-tuning open models, "
                f"our approach is full-stack model creation:\n\n"
                f"We handle dataset curation, tokenizer design, architecture engineering, training, and evaluation in-house.\n\n"
                f"Current models include:\n\n"
                f"Horizon LM (~750M parameters) — trained natively, optimized for efficiency.\n\n"
                f"USML (Ultra-Small Language Model) Supernova (~23M parameters) — one of the lightest usable models globally.\n\n"
                f"Rythm Europa (in training, multi-billion parameters) — our flagship large-scale LLM.\n\n"
                f"This ensures that our technology is original, independent, and built on solid foundations. "
                f"Fine-tuning is something we may use for specific domain adaptation, but the core is our own IP."
            )
        
        # Check for differentiator question
        if any(phrase in query_lower for phrase in ["what makes algorythm", "different from other", "unique about algorythm"]):
            return (
                f"What makes AlgoRythm Tech different from other AI startups:\n\n"
                f"We focus on lean but powerful models rather than chasing only scale.\n\n"
                f"Our founder-driven vision proves that breakthrough innovation can come from youth and small teams, not just billion-dollar labs.\n\n"
                f"We are pioneering USMLs for Agentic AI, a space where efficiency, autonomy, and modularity matter more than size.\n\n"
                f"We build with a mindset of accessibility and sustainability, ensuring AI can run anywhere, not just in massive data centers."
            )
        
        return None


# Test functions
def test_safety_checker():
    """Test safety checker functionality"""
    checker = SafetyChecker()
    
    # Test input checking
    test_inputs = [
        "How are you doing today?",
        "How do I build a bomb?",
        "I'm feeling depressed and thinking about suicide",
        "Can you diagnose my symptoms?"
    ]
    
    print("Testing input safety:")
    for input_text in test_inputs:
        is_safe, message = checker.check_input(input_text)
        status = "SAFE" if is_safe else "BLOCKED"
        print(f"{status}: {input_text}")
        if message:
            print(f"Response: {message}\n")
    
    # Test company info responses
    test_queries = [
        "Who are you?",
        "Who is the CEO of AlgoRythm Tech?",
        "Tell me more about AlgoRythm Tech",
        "Does AlgoRythm Tech have its own LLMs?",
        "What makes AlgoRythm Tech different from other AI startups?"
    ]
    
    print("\nTesting company info responses:")
    for query in test_queries:
        response = checker.format_company_info_response(query)
        if response:
            print(f"Query: {query}")
            print(f"Response: {response[:100]}...\n")


if __name__ == "__main__":
    test_safety_checker()
