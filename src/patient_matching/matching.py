"""
Patient Matching Algorithms

Implements deterministic and probabilistic matching:
- Deterministic: Exact matches on identifiers (MRN, SSN, DOB, phone, email)
- Probabilistic: Similarity algorithms (Jaro-Winkler, Levenshtein, phonetic)
- AI-Enhanced: OpenAI embeddings for semantic similarity
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

# Phonetic algorithms
import jellyfish

from .models import (
    Patient, MatchResult, MatchConfidence, 
    Identifier, IdentifierType, Address
)

logger = logging.getLogger(__name__)


@dataclass
class MatchWeights:
    """Configurable weights for match scoring"""
    # Deterministic weights
    enterprise_id: float = 1.0
    mrn: float = 0.8
    ssn: float = 0.9
    dob_exact: float = 0.35
    phone_exact: float = 0.3
    email_exact: float = 0.3
    
    # Probabilistic weights (for final score)
    deterministic_weight: float = 0.4
    name_weight: float = 0.35
    address_weight: float = 0.15
    embedding_weight: float = 0.1
    
    # Thresholds
    auto_merge_threshold: float = 0.85
    human_review_threshold: float = 0.65
    
    # Gender bonus
    gender_match_bonus: float = 0.05


class DeterministicMatcher:
    """
    Deterministic matching based on exact identifier matches
    
    High-confidence matches when strong identifiers match exactly.
    """
    
    def __init__(self, weights: MatchWeights = None):
        self.weights = weights or MatchWeights()
    
    def compute_score(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute deterministic match score between two patients
        
        Returns (score, details) where score is 0-1 and details 
        contains information about what matched.
        """
        score = 0.0
        details = {
            "matched_identifiers": [],
            "matched_contacts": [],
            "dob_match": False,
            "gender_match": False
        }
        
        # Check Enterprise ID match
        eid1 = patient1.get_enterprise_id()
        eid2 = patient2.get_enterprise_id()
        if eid1 and eid2 and eid1 == eid2:
            score = max(score, self.weights.enterprise_id)
            details["matched_identifiers"].append(("ENTERPRISE_ID", eid1))
        
        # Check SSN match
        ssn1 = patient1.get_ssn()
        ssn2 = patient2.get_ssn()
        if ssn1 and ssn2 and ssn1 == ssn2:
            score = max(score, self.weights.ssn)
            details["matched_identifiers"].append(("SSN", ssn1[-4:] + "****"))
        
        # Check MRN match (within same system)
        for id1 in patient1.identifiers:
            if id1.type == IdentifierType.MRN:
                for id2 in patient2.identifiers:
                    if id2.type == IdentifierType.MRN:
                        if id1.value == id2.value:
                            # MRN match - check if same system
                            if id1.system and id2.system and id1.system == id2.system:
                                score = max(score, self.weights.mrn)
                                details["matched_identifiers"].append(
                                    ("MRN", id1.value, id1.system)
                                )
                            elif not id1.system or not id2.system:
                                # Unknown systems - partial credit
                                score = max(score, self.weights.mrn * 0.7)
                                details["matched_identifiers"].append(
                                    ("MRN", id1.value, "unknown_system")
                                )
        
        # Check DOB match
        if patient1.birth_date and patient2.birth_date:
            if patient1.birth_date == patient2.birth_date:
                score += self.weights.dob_exact
                details["dob_match"] = True
        
        # Check phone match
        phone1 = patient1.get_primary_phone()
        phone2 = patient2.get_primary_phone()
        if phone1 and phone2 and phone1 == phone2:
            score += self.weights.phone_exact
            details["matched_contacts"].append(("phone", phone1[-4:] + "****"))
        
        # Check email match
        email1 = patient1.get_primary_email()
        email2 = patient2.get_primary_email()
        if email1 and email2 and email1 == email2:
            score += self.weights.email_exact
            details["matched_contacts"].append(("email", email1))
        
        # Gender match bonus
        if patient1.gender == patient2.gender:
            details["gender_match"] = True
            # Small bonus, not added to score directly but noted
        
        # Cap at 1.0
        score = min(1.0, score)
        
        return score, details


class ProbabilisticMatcher:
    """
    Probabilistic matching using similarity algorithms
    
    Computes similarity scores for names, addresses, and other fields
    using algorithms like Jaro-Winkler, Levenshtein, Soundex, etc.
    """
    
    def __init__(self, weights: MatchWeights = None):
        self.weights = weights or MatchWeights()
    
    def compute_name_similarity(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute name similarity using multiple algorithms
        
        Returns (score, details)
        """
        if not patient1.name or not patient2.name:
            return 0.0, {"reason": "missing_name"}
        
        details = {}
        scores = []
        
        # Full name Jaro-Winkler
        name1 = patient1.name.normalize()
        name2 = patient2.name.normalize()
        
        if name1 and name2:
            jw_score = jellyfish.jaro_winkler_similarity(name1, name2)
            details["full_name_jaro_winkler"] = jw_score
            scores.append(jw_score)
        
        # First name similarity
        first1 = (patient1.name.first_name or "").lower()
        first2 = (patient2.name.first_name or "").lower()
        
        if first1 and first2:
            first_jw = jellyfish.jaro_winkler_similarity(first1, first2)
            details["first_name_jaro_winkler"] = first_jw
            
            # Soundex match
            try:
                soundex1 = jellyfish.soundex(first1)
                soundex2 = jellyfish.soundex(first2)
                details["first_name_soundex_match"] = soundex1 == soundex2
            except:
                details["first_name_soundex_match"] = None
            
            # Metaphone match
            try:
                meta1 = jellyfish.metaphone(first1)
                meta2 = jellyfish.metaphone(first2)
                details["first_name_metaphone_match"] = meta1 == meta2
            except:
                details["first_name_metaphone_match"] = None
        
        # Last name similarity
        last1 = (patient1.name.last_name or "").lower()
        last2 = (patient2.name.last_name or "").lower()
        
        if last1 and last2:
            last_jw = jellyfish.jaro_winkler_similarity(last1, last2)
            details["last_name_jaro_winkler"] = last_jw
            
            # Soundex match
            try:
                soundex1 = jellyfish.soundex(last1)
                soundex2 = jellyfish.soundex(last2)
                details["last_name_soundex_match"] = soundex1 == soundex2
            except:
                details["last_name_soundex_match"] = None
            
            # If both first and last match well, boost score
            if first1 and first2:
                combined_score = (first_jw + last_jw) / 2
                scores.append(combined_score)
                details["combined_first_last"] = combined_score
        
        # Levenshtein distance (normalized)
        if name1 and name2:
            lev_dist = jellyfish.levenshtein_distance(name1, name2)
            max_len = max(len(name1), len(name2))
            lev_score = 1 - (lev_dist / max_len) if max_len > 0 else 0
            details["levenshtein_normalized"] = lev_score
            scores.append(lev_score)
        
        # Final score is weighted average favoring Jaro-Winkler
        if scores:
            # Give more weight to Jaro-Winkler (first score)
            if len(scores) == 1:
                final_score = scores[0]
            else:
                final_score = scores[0] * 0.6 + sum(scores[1:]) / len(scores[1:]) * 0.4
        else:
            final_score = 0.0
        
        details["final_score"] = final_score
        return final_score, details
    
    def compute_address_similarity(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute address similarity
        
        Uses normalized addresses and token-based comparison
        """
        addr1 = patient1.get_primary_address()
        addr2 = patient2.get_primary_address()
        
        if not addr1 or not addr2:
            return 0.0, {"reason": "missing_address"}
        
        details = {}
        
        # Normalize addresses
        norm1 = addr1.normalize()
        norm2 = addr2.normalize()
        
        details["normalized_addr1"] = norm1
        details["normalized_addr2"] = norm2
        
        # Jaro-Winkler on full normalized address
        jw_score = jellyfish.jaro_winkler_similarity(norm1, norm2)
        details["jaro_winkler"] = jw_score
        
        # Token-based comparison
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        
        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            jaccard = len(intersection) / len(union) if union else 0
            details["token_jaccard"] = jaccard
            details["shared_tokens"] = list(intersection)
        else:
            jaccard = 0
            details["token_jaccard"] = 0
        
        # Postal code exact match bonus
        if addr1.postal_code and addr2.postal_code:
            postal_match = addr1.postal_code == addr2.postal_code
            details["postal_code_match"] = postal_match
            if postal_match:
                # Boost score for postal code match
                jw_score = min(1.0, jw_score + 0.1)
        
        # City exact match bonus
        if addr1.city and addr2.city:
            city_match = addr1.city.lower() == addr2.city.lower()
            details["city_match"] = city_match
        
        # State exact match bonus
        if addr1.state and addr2.state:
            state_match = addr1.state.lower() == addr2.state.lower()
            details["state_match"] = state_match
        
        # Combine scores
        final_score = (jw_score * 0.7) + (jaccard * 0.3)
        details["final_score"] = final_score
        
        return final_score, details
    
    def compute_dob_similarity(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute DOB similarity with transposition tolerance
        
        Handles typos like swapped month/day
        """
        if not patient1.birth_date or not patient2.birth_date:
            return 0.0, {"reason": "missing_dob"}
        
        dob1 = patient1.birth_date
        dob2 = patient2.birth_date
        
        details = {
            "dob1": dob1.isoformat(),
            "dob2": dob2.isoformat()
        }
        
        # Exact match
        if dob1 == dob2:
            return 1.0, {"exact_match": True, **details}
        
        # Check for transposed month/day
        if (dob1.year == dob2.year and 
            dob1.month == dob2.day and 
            dob1.day == dob2.month):
            details["transposition_detected"] = True
            return 0.8, details
        
        # Same year
        if dob1.year == dob2.year:
            details["year_match"] = True
            # Check if close in days
            day_diff = abs((dob1 - dob2).days)
            details["day_difference"] = day_diff
            
            if day_diff <= 1:
                return 0.9, details  # Off by one day
            elif day_diff <= 30:
                return 0.5, details  # Within a month
            else:
                return 0.2, details  # Same year only
        
        # Different years - very low similarity
        year_diff = abs(dob1.year - dob2.year)
        details["year_difference"] = year_diff
        
        if year_diff == 1:
            # Could be typo
            return 0.1, details
        
        return 0.0, details


class LLMMatcher:
    """
    AI-Enhanced matching using GPT models (GPT-4, GPT-4o, etc.)
    
    Uses large language models to:
    1. Analyze patient record similarities
    2. Identify potential typos and variations
    3. Provide reasoning for match decisions
    4. Handle complex matching scenarios
    
    Supports:
    - OpenAI API (GPT-4, GPT-4o, GPT-4-turbo)
    - Azure OpenAI (deployed GPT-4 models)
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        # Azure OpenAI configuration
        use_azure: bool = False,
        azure_endpoint: str = None,
        azure_deployment: str = None,
        azure_api_version: str = "2024-02-01",
        use_azure_ad: bool = True  # Use Azure AD auth by default for Azure OpenAI
    ):
        """
        Initialize LLM matcher
        
        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model name (gpt-4, gpt-4o, gpt-4-turbo)
            use_azure: Use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure OpenAI deployment name
            azure_api_version: Azure OpenAI API version
            use_azure_ad: Use Azure AD authentication (DefaultAzureCredential) instead of API key
        """
        import os
        
        self.use_azure = use_azure
        self.use_azure_ad = use_azure_ad
        self.model = model
        self._client = None
        
        if use_azure:
            self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", model)
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def client(self):
        """Lazy initialization of OpenAI/Azure OpenAI client"""
        if self._client is None:
            try:
                if self.use_azure:
                    from openai import AzureOpenAI
                    
                    if self.use_azure_ad and not self.api_key:
                        # Use Azure AD authentication with DefaultAzureCredential
                        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                        credential = DefaultAzureCredential()
                        token_provider = get_bearer_token_provider(
                            credential, "https://cognitiveservices.azure.com/.default"
                        )
                        self._client = AzureOpenAI(
                            azure_ad_token_provider=token_provider,
                            azure_endpoint=self.azure_endpoint,
                            api_version=self.azure_api_version
                        )
                        logger.info(f"Using Azure OpenAI GPT endpoint with Azure AD auth: {self.azure_endpoint}")
                    else:
                        # Use API key authentication
                        self._client = AzureOpenAI(
                            api_key=self.api_key,
                            azure_endpoint=self.azure_endpoint,
                            api_version=self.azure_api_version
                        )
                        logger.info(f"Using Azure OpenAI GPT endpoint with API key: {self.azure_endpoint}")
                else:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.api_key)
                    logger.info(f"Using OpenAI API with model: {self.model}")
            except ImportError as e:
                raise ImportError(
                    "openai and azure-identity packages required. Install with: pip install openai azure-identity"
                )
        return self._client
    
    def analyze_match(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Use GPT to analyze if two patients are likely the same person
        
        Returns (score, details) with reasoning
        """
        prompt = self._build_match_prompt(patient1, patient2)
        
        try:
            deployment = self.azure_deployment if self.use_azure else self.model
            
            response = self.client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a patient identity matching expert. Analyze two patient records and determine if they likely represent the same person.

Consider:
- Name variations (nicknames, typos, maiden names)
- Date of birth (transpositions, typos)
- Address similarities (moved, abbreviations)
- Phone/email matches
- Identifier matches

Respond in JSON format:
{
    "match_score": 0.0-1.0,
    "confidence": "high|medium|low",
    "reasoning": "explanation",
    "name_analysis": "analysis of name similarities",
    "potential_issues": ["list of concerns"],
    "recommendation": "merge|review|no_match"
}"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            import json
            result = json.loads(result_text)
            
            score = float(result.get("match_score", 0.0))
            details = {
                "llm_confidence": result.get("confidence", "low"),
                "llm_reasoning": result.get("reasoning", ""),
                "name_analysis": result.get("name_analysis", ""),
                "potential_issues": result.get("potential_issues", []),
                "recommendation": result.get("recommendation", "no_match"),
                "model_used": deployment
            }
            
            return score, details
            
        except Exception as e:
            logger.error(f"LLM match analysis failed: {e}")
            return 0.0, {"error": str(e)}
    
    def _build_match_prompt(self, patient1: Patient, patient2: Patient) -> str:
        """Build prompt for patient comparison"""
        def format_patient(p: Patient, label: str) -> str:
            lines = [f"**{label}:**"]
            if p.name:
                lines.append(f"  Name: {p.name.full_name}")
            if p.birth_date:
                lines.append(f"  DOB: {p.birth_date.isoformat()}")
            if p.gender:
                lines.append(f"  Gender: {p.gender.value}")
            if p.identifiers:
                ids = [f"{i.type.value}:{i.value}" for i in p.identifiers[:5]]
                lines.append(f"  Identifiers: {', '.join(ids)}")
            if p.addresses:
                addr = p.addresses[0]
                lines.append(f"  Address: {addr.full_address}")
            if p.contact_points:
                contacts = [f"{c.system}:{c.value}" for c in p.contact_points[:3]]
                lines.append(f"  Contacts: {', '.join(contacts)}")
            return "\n".join(lines)
        
        return f"""Compare these two patient records:

{format_patient(patient1, "Patient 1")}

{format_patient(patient2, "Patient 2")}

Are these the same person? Analyze the evidence and provide your assessment."""


class EmbeddingMatcher:
    """
    AI-Enhanced matching using OpenAI or Azure OpenAI embeddings
    
    Generates embeddings for patient profiles and computes
    cosine similarity for semantic matching.
    
    Supports:
    - OpenAI API (default)
    - Azure OpenAI (set use_azure=True)
    - Azure AD authentication (DefaultAzureCredential)
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-3-small",
        # Azure OpenAI configuration
        use_azure: bool = False,
        azure_endpoint: str = None,
        azure_deployment: str = None,
        azure_api_version: str = "2024-02-01",
        use_azure_ad: bool = True  # Use Azure AD auth by default for Azure OpenAI
    ):
        """
        Initialize embedding matcher
        
        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model name (for OpenAI) or deployment name (for Azure)
            use_azure: Use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
            azure_deployment: Azure OpenAI deployment name for embeddings
            azure_api_version: Azure OpenAI API version
            use_azure_ad: Use Azure AD authentication (DefaultAzureCredential) instead of API key
        """
        import os
        
        self.use_azure = use_azure
        self.use_azure_ad = use_azure_ad
        self.model = model
        self._client = None
        
        if use_azure:
            self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", model)
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def client(self):
        """Lazy initialization of OpenAI/Azure OpenAI client"""
        if self._client is None:
            try:
                if self.use_azure:
                    from openai import AzureOpenAI
                    
                    if self.use_azure_ad and not self.api_key:
                        # Use Azure AD authentication with DefaultAzureCredential
                        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                        credential = DefaultAzureCredential()
                        token_provider = get_bearer_token_provider(
                            credential, "https://cognitiveservices.azure.com/.default"
                        )
                        self._client = AzureOpenAI(
                            azure_ad_token_provider=token_provider,
                            azure_endpoint=self.azure_endpoint,
                            api_version=self.azure_api_version
                        )
                        logger.info(f"Using Azure OpenAI endpoint with Azure AD auth: {self.azure_endpoint}")
                    else:
                        # Use API key authentication
                        self._client = AzureOpenAI(
                            api_key=self.api_key,
                            azure_endpoint=self.azure_endpoint,
                            api_version=self.azure_api_version
                        )
                        logger.info(f"Using Azure OpenAI endpoint with API key: {self.azure_endpoint}")
                else:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.api_key)
                    logger.info("Using OpenAI API")
            except ImportError as e:
                raise ImportError(
                    "openai and azure-identity packages required. Install with: pip install openai azure-identity"
                )
        return self._client
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        if not text:
            return []
        
        try:
            if self.use_azure:
                # Azure OpenAI uses deployment name
                response = self.client.embeddings.create(
                    model=self.azure_deployment,
                    input=text
                )
            else:
                # Standard OpenAI uses model name
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def get_patient_embedding(self, patient: Patient) -> List[float]:
        """
        Generate embedding for patient profile
        
        If patient already has embedding, return it.
        Otherwise generate new one.
        """
        if patient.embedding:
            return patient.embedding
        
        profile_text = patient.to_profile_text()
        return self.get_embedding(profile_text)
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_similarity(
        self,
        patient1: Patient,
        patient2: Patient
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute embedding similarity between two patients
        
        Returns (score, details)
        """
        details = {}
        
        emb1 = self.get_patient_embedding(patient1)
        emb2 = self.get_patient_embedding(patient2)
        
        if not emb1 or not emb2:
            return 0.0, {"reason": "embedding_failed"}
        
        similarity = self.cosine_similarity(emb1, emb2)
        details["cosine_similarity"] = similarity
        details["embedding_dimension"] = len(emb1)
        
        return similarity, details


class PatientMatcher:
    """
    Main patient matching service
    
    Combines deterministic, probabilistic, and AI-enhanced matching
    to produce final match scores and recommendations.
    
    Supports:
    - OpenAI (GPT-4, GPT-4o, embeddings)
    - Azure OpenAI (GPT-4, GPT-4o, embeddings)
    
    AI Matching Modes:
    - Embeddings: Fast semantic similarity using embedding models
    - LLM: Deep analysis using GPT-4/GPT-4o with reasoning
    """
    
    def __init__(
        self,
        weights: MatchWeights = None,
        # Embedding-based matching
        use_embeddings: bool = False,
        # LLM-based matching (GPT-4, GPT-4o)
        use_llm: bool = False,
        llm_model: str = "gpt-4o",
        # OpenAI configuration
        openai_api_key: str = None,
        # Azure OpenAI configuration
        use_azure_openai: bool = False,
        azure_openai_endpoint: str = None,
        azure_openai_key: str = None,
        azure_openai_embedding_deployment: str = None,
        azure_openai_gpt_deployment: str = None,
        azure_openai_api_version: str = "2024-02-01"
    ):
        """
        Initialize patient matcher
        
        Args:
            weights: Match scoring weights
            use_embeddings: Enable embedding-based semantic matching
            use_llm: Enable LLM-based deep analysis (GPT-4/GPT-4o)
            llm_model: LLM model to use (gpt-4, gpt-4o, gpt-4-turbo)
            openai_api_key: OpenAI API key (for standard OpenAI)
            use_azure_openai: Use Azure OpenAI instead of OpenAI
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_key: Azure OpenAI API key
            azure_openai_embedding_deployment: Azure deployment for embeddings
            azure_openai_gpt_deployment: Azure deployment for GPT model
            azure_openai_api_version: Azure OpenAI API version
        """
        self.weights = weights or MatchWeights()
        self.deterministic = DeterministicMatcher(self.weights)
        self.probabilistic = ProbabilisticMatcher(self.weights)
        
        # Initialize embedding matcher
        self.use_embeddings = use_embeddings
        self.embedding_matcher = None
        if use_embeddings:
            try:
                if use_azure_openai:
                    self.embedding_matcher = EmbeddingMatcher(
                        api_key=azure_openai_key,
                        use_azure=True,
                        azure_endpoint=azure_openai_endpoint,
                        azure_deployment=azure_openai_embedding_deployment,
                        azure_api_version=azure_openai_api_version
                    )
                    logger.info("Using Azure OpenAI for embeddings")
                else:
                    self.embedding_matcher = EmbeddingMatcher(api_key=openai_api_key)
                    logger.info("Using OpenAI for embeddings")
            except Exception as e:
                logger.warning(f"Embedding matcher not available: {e}")
                self.use_embeddings = False
        
        # Initialize LLM matcher (GPT-4/GPT-4o)
        self.use_llm = use_llm
        self.llm_matcher = None
        if use_llm:
            try:
                if use_azure_openai:
                    self.llm_matcher = LLMMatcher(
                        api_key=azure_openai_key,
                        model=llm_model,
                        use_azure=True,
                        azure_endpoint=azure_openai_endpoint,
                        azure_deployment=azure_openai_gpt_deployment,
                        azure_api_version=azure_openai_api_version
                    )
                    logger.info(f"Using Azure OpenAI GPT ({azure_openai_gpt_deployment}) for LLM matching")
                else:
                    self.llm_matcher = LLMMatcher(
                        api_key=openai_api_key,
                        model=llm_model
                    )
                    logger.info(f"Using OpenAI {llm_model} for LLM matching")
            except Exception as e:
                logger.warning(f"LLM matcher not available: {e}")
                self.use_llm = False
    
    def match(
        self,
        patient1: Patient,
        patient2: Patient,
        compute_embedding: bool = None,
        use_llm_analysis: bool = None
    ) -> MatchResult:
        """
        Compute comprehensive match score between two patients
        
        Args:
            patient1: First patient
            patient2: Second patient
            compute_embedding: Override for embedding computation
            use_llm_analysis: Override for LLM-based analysis
        
        Returns:
            MatchResult with scores and confidence level
        """
        # Deterministic score
        det_score, det_details = self.deterministic.compute_score(
            patient1, patient2
        )
        
        # Probabilistic name similarity
        name_score, name_details = self.probabilistic.compute_name_similarity(
            patient1, patient2
        )
        
        # Probabilistic address similarity
        addr_score, addr_details = self.probabilistic.compute_address_similarity(
            patient1, patient2
        )
        
        # DOB similarity (for additional context)
        dob_score, dob_details = self.probabilistic.compute_dob_similarity(
            patient1, patient2
        )
        
        # Embedding similarity (optional)
        embedding_score = 0.0
        embedding_details = {}
        
        use_emb = compute_embedding if compute_embedding is not None else self.use_embeddings
        if use_emb and self.embedding_matcher:
            embedding_score, embedding_details = self.embedding_matcher.compute_similarity(
                patient1, patient2
            )
        
        # Compile shared identifiers
        shared_ids = []
        for ident_info in det_details.get("matched_identifiers", []):
            if len(ident_info) >= 2:
                shared_ids.append(f"{ident_info[0]}:{ident_info[1]}")
        
        # LLM analysis (optional - for human review cases or complex matches)
        llm_score = 0.0
        llm_details = {}
        
        do_llm = use_llm_analysis if use_llm_analysis is not None else self.use_llm
        if do_llm and self.llm_matcher:
            llm_score, llm_details = self.llm_matcher.analyze_match(patient1, patient2)
        
        # Compile all match details
        match_details = {
            "deterministic": det_details,
            "name": name_details,
            "address": addr_details,
            "dob": dob_details,
            "embedding": embedding_details,
            "llm": llm_details
        }
        
        # If LLM analysis was used, blend it with traditional scoring
        if do_llm and self.llm_matcher and llm_score > 0:
            # LLM score can override or supplement traditional scoring
            # Use LLM as a weighted component (20% weight when enabled)
            llm_weight = 0.2
            traditional_weight = 1.0 - llm_weight
            
            # Compute traditional score first
            result = MatchResult.from_scores(
                patient1_id=patient1.id,
                patient2_id=patient2.id,
                deterministic_score=det_score,
                name_similarity=name_score,
                address_similarity=addr_score,
                embedding_similarity=embedding_score,
                shared_identifiers=shared_ids,
                match_details=match_details
            )
            
            # Blend with LLM score
            blended_score = (result.score * traditional_weight) + (llm_score * llm_weight)
            result.score = blended_score
            result.match_details["llm_blended"] = True
            result.match_details["llm_score"] = llm_score
            result.match_details["llm_recommendation"] = llm_details.get("recommendation", "unknown")
        else:
            # Create match result with weighted scoring (no LLM)
            result = MatchResult.from_scores(
                patient1_id=patient1.id,
                patient2_id=patient2.id,
                deterministic_score=det_score,
                name_similarity=name_score,
                address_similarity=addr_score,
                embedding_similarity=embedding_score,
                shared_identifiers=shared_ids,
                match_details=match_details
            )
        
        # Apply gender bonus if applicable
        if det_details.get("gender_match"):
            result.score = min(1.0, result.score + self.weights.gender_match_bonus)
            result.match_details["gender_bonus_applied"] = True
        
        # Recalculate confidence after any adjustments
        if result.score >= self.weights.auto_merge_threshold:
            result.confidence = MatchConfidence.AUTO_MERGE
        elif result.score >= self.weights.human_review_threshold:
            result.confidence = MatchConfidence.HUMAN_REVIEW
        else:
            result.confidence = MatchConfidence.NO_MATCH
        
        logger.debug(
            f"Match score for {patient1.id} vs {patient2.id}: "
            f"{result.score:.3f} ({result.confidence.value})"
        )
        
        return result
    
    def find_matches(
        self,
        patient: Patient,
        candidates: List[Patient],
        min_score: float = 0.3,
        compute_embeddings: bool = False
    ) -> List[MatchResult]:
        """
        Find matches for a patient among a list of candidates
        
        Args:
            patient: Patient to match
            candidates: List of candidate patients
            min_score: Minimum score to include in results
            compute_embeddings: Whether to compute embeddings
        
        Returns:
            List of MatchResults sorted by score descending
        """
        results = []
        
        for candidate in candidates:
            if candidate.id == patient.id:
                continue
            
            result = self.match(
                patient, 
                candidate,
                compute_embedding=compute_embeddings
            )
            
            if result.score >= min_score:
                results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def batch_match(
        self,
        patients: List[Patient],
        min_score: float = 0.3
    ) -> List[MatchResult]:
        """
        Find all potential matches within a batch of patients
        
        Uses pairwise comparison with early exit for efficiency.
        
        Args:
            patients: List of patients to compare
            min_score: Minimum score threshold
        
        Returns:
            List of MatchResults for all pairs above threshold
        """
        results = []
        n = len(patients)
        
        for i in range(n):
            for j in range(i + 1, n):
                result = self.match(
                    patients[i],
                    patients[j],
                    compute_embedding=False  # Skip embedding for batch
                )
                
                if result.score >= min_score:
                    results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(
            f"Batch match completed: {n} patients, "
            f"{len(results)} potential matches found"
        )
        
        return results
