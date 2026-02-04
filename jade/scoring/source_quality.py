"""
Source Quality Scoring Module.

This module evaluates the quality and credibility of reference sources
used in verification results. It considers:

1. Source Tier: Official (T1) > Authoritative (T2) > Professional (T3) > Unknown (T4)

Example:
    >>> from jade.scoring.source_quality import SourceQualityScorer
    >>> scorer = SourceQualityScorer()
    >>> result = scorer.evaluate_sources(
    ...     query="What is TikTok's monthly active users in 2024?",
    ...     reference_urls=["https://newsroom.tiktok.com/en-us/stats", "https://statista.com/..."]
    ... )
    >>> print(f"Quality score: {result.overall_score:.2f}")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Literal
from urllib.parse import urlparse
from enum import Enum


class SourceTier(str, Enum):
    """Source credibility tiers."""
    TIER_1 = "T1"  # Official/First-party (highest credibility)
    TIER_2 = "T2"  # Authoritative third-party
    TIER_3 = "T3"  # Professional/known sources
    TIER_4 = "T4"  # Unknown/small sources (lowest credibility)


# =============================================================================
# Source Classification Data
# =============================================================================

# Tier 1: Official platforms and government sources
# These are first-party sources - the platforms themselves
TIER_1_DOMAINS: Dict[str, List[str]] = {
    # Major tech platforms (official domains)
    "tiktok": ["tiktok.com", "newsroom.tiktok.com", "bytedance.com"],
    "amazon": ["amazon.com", "aboutamazon.com", "aws.amazon.com", "amazon.jobs"],
    "google": ["google.com", "blog.google", "cloud.google.com", "developers.google.com"],
    "meta": ["meta.com", "facebook.com", "fb.com", "about.fb.com", "instagram.com"],
    "apple": ["apple.com", "developer.apple.com"],
    "microsoft": ["microsoft.com", "azure.microsoft.com", "news.microsoft.com"],
    "twitter": ["twitter.com", "x.com", "about.twitter.com"],
    "youtube": ["youtube.com", "blog.youtube"],
    "netflix": ["netflix.com", "about.netflix.com", "ir.netflix.com"],
    "spotify": ["spotify.com", "newsroom.spotify.com"],
    "uber": ["uber.com", "newsroom.uber.com"],
    "airbnb": ["airbnb.com", "news.airbnb.com"],
    "linkedin": ["linkedin.com", "news.linkedin.com", "about.linkedin.com"],
    "snapchat": ["snapchat.com", "snap.com", "newsroom.snap.com"],
    "pinterest": ["pinterest.com", "newsroom.pinterest.com"],
    "shopify": ["shopify.com", "news.shopify.com"],
    "alibaba": ["alibaba.com", "alibabagroup.com"],
    "tencent": ["tencent.com", "qq.com", "wechat.com"],
    "baidu": ["baidu.com"],
    "jd": ["jd.com", "jdcorporateblog.com"],
    
    # E-commerce specific
    "ebay": ["ebay.com", "ebayinc.com"],
    "walmart": ["walmart.com", "corporate.walmart.com"],
    "target": ["target.com", "corporate.target.com"],
    "etsy": ["etsy.com", "investors.etsy.com"],
    
    # Finance
    "paypal": ["paypal.com", "newsroom.paypal.com"],
    "stripe": ["stripe.com"],
    "square": ["squareup.com", "block.xyz"],
    
    # Gaming
    "steam": ["steampowered.com", "store.steampowered.com"],
    "epic": ["epicgames.com", "unrealengine.com"],
    "roblox": ["roblox.com", "corp.roblox.com"],
    
    # Government (generic pattern)
    "government": [],  # Handled by .gov, .gov.cn, etc. pattern
}

# Tier 1: Government domain patterns
TIER_1_GOVERNMENT_PATTERNS = [
    r"\.gov$",
    r"\.gov\.[a-z]{2}$",  # .gov.uk, .gov.cn, etc.
    r"\.mil$",
    r"\.edu$",
    r"\.ac\.[a-z]{2}$",  # Academic domains like .ac.uk
]

# Tier 2: Authoritative third-party sources
TIER_2_DOMAINS: Set[str] = {
    # Data & Analytics platforms
    "statista.com",
    "similarweb.com",
    "semrush.com",
    "alexa.com",
    "comscore.com",
    "appannie.com",
    "data.ai",
    "sensor tower.com",
    "apptopia.com",
    "emarketer.com",
    "insiderintelligence.com",
    
    # Research & Consulting
    "gartner.com",
    "forrester.com",
    "mckinsey.com",
    "bcg.com",
    "bain.com",
    "deloitte.com",
    "pwc.com",
    "kpmg.com",
    "ey.com",
    "accenture.com",
    "idc.com",
    "nielsen.com",
    "kantar.com",
    "ipsos.com",
    "gallup.com",
    "pewresearch.org",
    
    # Major news & financial media
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnbc.com",
    "forbes.com",
    "fortune.com",
    "businessinsider.com",
    "bbc.com",
    "bbc.co.uk",
    "cnn.com",
    "theguardian.com",
    "apnews.com",
    
    # Academic / Scientific
    "arxiv.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "ieee.org",
    "acm.org",
    "researchgate.net",
    "scholar.google.com",
    
    # Wikipedia (generally reliable for factual claims)
    "wikipedia.org",
    "en.wikipedia.org",
}

# Tier 3: Professional / Known sources
TIER_3_DOMAINS: Set[str] = {
    # Tech media
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
    "engadget.com",
    "cnet.com",
    "zdnet.com",
    "venturebeat.com",
    "thenextweb.com",
    "mashable.com",
    "gizmodo.com",
    "9to5mac.com",
    "9to5google.com",
    "androidcentral.com",
    "macrumors.com",
    "tomsguide.com",
    "howtogeek.com",
    "digitaltrends.com",
    "pcmag.com",
    "theinformation.com",
    
    # Developer / Tech communities
    "github.com",
    "stackoverflow.com",
    "dev.to",
    "hackernews.com",
    "news.ycombinator.com",
    "producthunt.com",
    "reddit.com",
    
    # Business / Startup
    "crunchbase.com",
    "pitchbook.com",
    "cbinsights.com",
    "techstars.com",
    "ycombinator.com",
    
    # Blogging platforms (variable quality but recognized)
    "medium.com",
    "substack.com",
    "wordpress.com",
    "hubspot.com",
    "quora.com",
    
    # Review sites
    "g2.com",
    "capterra.com",
    "trustpilot.com",
    "glassdoor.com",
}

# Tier scores
TIER_SCORES = {
    SourceTier.TIER_1: 1.0,
    SourceTier.TIER_2: 0.75,
    SourceTier.TIER_3: 0.5,
    SourceTier.TIER_4: 0.25,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SourceEvaluation:
    """Evaluation result for a single source URL."""
    url: str
    domain: str
    tier: SourceTier
    tier_score: float
    
    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "tier": self.tier.value,
            "tier_score": self.tier_score,
        }


@dataclass
class SourceQualityResult:
    """Result of source quality evaluation."""
    
    # Overall metrics
    overall_score: float  # Weighted average of source scores (0.0 - 1.0)
    source_count: int
    
    # Tier distribution
    tier_distribution: Dict[str, int]  # {"T1": 2, "T2": 1, ...}
    
    # Individual source evaluations
    source_evaluations: List[SourceEvaluation] = field(default_factory=list)
    
    # Quality flags
    has_official_source: bool = False  # At least one T1 source
    all_unknown_sources: bool = False  # All sources are T4
    
    # Recommendations
    quality_grade: Literal["A", "B", "C", "D", "F"] = "C"
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "source_count": self.source_count,
            "tier_distribution": self.tier_distribution,
            "has_official_source": self.has_official_source,
            "all_unknown_sources": self.all_unknown_sources,
            "quality_grade": self.quality_grade,
            "recommendations": self.recommendations,
            "source_evaluations": [s.to_dict() for s in self.source_evaluations],
        }


# =============================================================================
# Source Quality Scorer
# =============================================================================

class SourceQualityScorer:
    """
    Scorer for evaluating the quality and credibility of reference sources.
    
    Implements a tiered scoring system:
    - T1 (Official): First-party platform data, government sources
    - T2 (Authoritative): Major data platforms, research firms, mainstream media
    - T3 (Professional): Tech media, developer communities
    - T4 (Unknown): Unrecognized sources
    
    Example:
        >>> scorer = SourceQualityScorer()
        >>> result = scorer.evaluate_sources(
        ...     query="TikTok's user growth in 2024",
        ...     reference_urls=["https://newsroom.tiktok.com/...", "https://statista.com/..."]
        ... )
        >>> print(result.overall_score)  # Higher because of first-party TikTok source
    """
    
    def __init__(
        self,
        tier_1_domains: Optional[Dict[str, List[str]]] = None,
        tier_2_domains: Optional[Set[str]] = None,
        tier_3_domains: Optional[Set[str]] = None,
        single_source_penalty: float = 0.9,
        no_verification_penalty: float = 0.85,
    ):
        """
        Initialize the source quality scorer.
        
        Uses penalty-based scoring (NOT bonus-based):
        - Baseline: T1 sources = 1.0 (standard, no penalty)
        - Lower tiers get inherent penalties (T2=0.75, T3=0.5, T4=0.25)
        - Additional penalties for single source or no verification
        
        Args:
            tier_1_domains: Custom T1 domain mapping (platform -> domains)
            tier_2_domains: Custom T2 domain set
            tier_3_domains: Custom T3 domain set
            single_source_penalty: Multiplier when only 1 source (< 1.0, default 0.9)
            no_verification_penalty: Multiplier when no verification (< 1.0, default 0.85)
        """
        self.tier_1_domains = tier_1_domains or TIER_1_DOMAINS
        self.tier_2_domains = tier_2_domains or TIER_2_DOMAINS
        self.tier_3_domains = tier_3_domains or TIER_3_DOMAINS
        self.single_source_penalty = single_source_penalty
        self.no_verification_penalty = no_verification_penalty
        # Flatten Tier-1 domains for fast membership checks
        self._tier_1_domains_flat: Set[str] = set()
        for _, domains in self.tier_1_domains.items():
            for domain in domains:
                self._tier_1_domains_flat.add(domain.lower())
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def _is_government_domain(self, domain: str) -> bool:
        """Check if domain is a government domain."""
        for pattern in TIER_1_GOVERNMENT_PATTERNS:
            if re.search(pattern, domain):
                return True
        return False
    
    def _classify_source(
        self, 
        url: str,
    ) -> SourceEvaluation:
        """
        Classify a single source URL.
        
        Returns SourceEvaluation with tier and tier_score.
        """
        domain = self._extract_domain(url)
        
        if not domain:
            return SourceEvaluation(
                url=url,
                domain="",
                tier=SourceTier.TIER_4,
                tier_score=TIER_SCORES[SourceTier.TIER_4],
            )
        
        # Check Tier 1 (Official sources)
        tier = SourceTier.TIER_4
        
        # Check if domain matches any T1 platform
        if domain in self._tier_1_domains_flat:
            tier = SourceTier.TIER_1
        else:
            # Check subdomain matches (e.g., newsroom.tiktok.com -> tiktok.com)
            for t1_domain in self._tier_1_domains_flat:
                if domain == t1_domain or domain.endswith("." + t1_domain):
                    tier = SourceTier.TIER_1
                    break
        
        # Check government domains
        if tier == SourceTier.TIER_4 and self._is_government_domain(domain):
            tier = SourceTier.TIER_1
        
        # Check Tier 2
        if tier == SourceTier.TIER_4:
            for t2_domain in self.tier_2_domains:
                if domain == t2_domain or domain.endswith("." + t2_domain):
                    tier = SourceTier.TIER_2
                    break
        
        # Check Tier 3
        if tier == SourceTier.TIER_4:
            for t3_domain in self.tier_3_domains:
                if domain == t3_domain or domain.endswith("." + t3_domain):
                    tier = SourceTier.TIER_3
                    break
        
        # Calculate scores
        tier_score = TIER_SCORES[tier]
        
        return SourceEvaluation(
            url=url,
            domain=domain,
            tier=tier,
            tier_score=tier_score,
        )
    
    def _calculate_quality_grade(
        self,
        overall_score: float,
        has_official: bool,
        all_unknown: bool,
    ) -> Literal["A", "B", "C", "D", "F"]:
        """Calculate quality grade based on metrics."""
        if all_unknown:
            return "F"
        if overall_score >= 0.75 and has_official:
            return "A"
        elif overall_score >= 0.7:
            return "B"
        elif overall_score >= 0.5:
            return "C"
        elif overall_score >= 0.3:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(
        self,
        result: SourceQualityResult
    ) -> List[str]:
        """Generate improvement recommendations."""
        recs = []
        
        if result.all_unknown_sources:
            recs.append("All sources are from unknown/unrecognized domains. Consider using more authoritative sources.")
        
        if not result.has_official_source:
            recs.append("No official (T1) sources found. Adding government data or platform-official sources would increase credibility.")
        
        t4_count = result.tier_distribution.get("T4", 0)
        if t4_count > result.source_count // 2:
            recs.append(f"More than half of sources ({t4_count}/{result.source_count}) are from unknown domains. "
                       f"Consider replacing with recognized authoritative sources.")
        
        return recs
    
    def evaluate_sources(
        self,
        query: str,
        reference_urls: List[str],
        weight_by_position: bool = False,
    ) -> SourceQualityResult:
        """
        Evaluate the quality of reference sources.
        
        Args:
            query: The original query (used to detect mentioned platforms)
            reference_urls: List of reference URLs to evaluate
            weight_by_position: If True, earlier URLs get higher weight (first = most important)
        
        Returns:
            SourceQualityResult with overall score and detailed breakdown
        """
        # Handle empty input
        if not reference_urls:
            return SourceQualityResult(
                overall_score=0.0,
                source_count=0,
                tier_distribution={},
                all_unknown_sources=True,
                quality_grade="F",
                recommendations=["No reference sources provided."],
            )
        
        # Evaluate each source
        evaluations = []
        for url in reference_urls:
            if url and url.strip():
                eval_result = self._classify_source(url)
                evaluations.append(eval_result)
        
        if not evaluations:
            return SourceQualityResult(
                overall_score=0.0,
                source_count=0,
                tier_distribution={},
                all_unknown_sources=True,
                quality_grade="F",
                recommendations=["No valid reference URLs provided."],
            )
        
        # Calculate tier distribution
        tier_dist = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
        for eval_result in evaluations:
            tier_dist[eval_result.tier.value] += 1
        
        # Calculate overall score (tier-only; no bonuses)
        if weight_by_position:
            # Earlier sources weighted higher (first source = 1.0, decreasing)
            total_weight = 0.0
            weighted_sum = 0.0
            for i, eval_result in enumerate(evaluations):
                weight = 1.0 / (i + 1)  # 1, 0.5, 0.33, 0.25, ...
                weighted_sum += eval_result.tier_score * weight
                total_weight += weight
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            # Simple average
            overall_score = sum(e.tier_score for e in evaluations) / len(evaluations)
        
        # Flags
        has_official = tier_dist["T1"] > 0
        all_unknown = all(e.tier == SourceTier.TIER_4 for e in evaluations)
        
        # Build result
        result = SourceQualityResult(
            overall_score=round(overall_score, 4),
            source_count=len(evaluations),
            tier_distribution=tier_dist,
            source_evaluations=evaluations,
            has_official_source=has_official,
            all_unknown_sources=all_unknown,
        )
        
        # Calculate grade and recommendations
        result.quality_grade = self._calculate_quality_grade(
            overall_score, has_official, all_unknown
        )
        result.recommendations = self._generate_recommendations(result)
        
        return result
    
    def evaluate_verification_results(
        self,
        query: str,
        verification_results: List[dict],
    ) -> SourceQualityResult:
        """
        Evaluate source quality from verification results.
        
        Convenience method that extracts reference_urls from verification results.
        
        Args:
            query: Original query
            verification_results: List of verification result dicts with 'reference_urls' field
        
        Returns:
            SourceQualityResult
        """
        all_urls = []
        for result in verification_results:
            ref_urls = result.get("reference_urls", {})
            if isinstance(ref_urls, dict):
                all_urls.extend(ref_urls.get("supporting", []))
                all_urls.extend(ref_urls.get("contradicting", []))
            elif isinstance(ref_urls, list):
                all_urls.extend(ref_urls)
            
            # Also check source_url field
            source_url = result.get("source_url")
            if source_url:
                all_urls.append(source_url)
        
        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return self.evaluate_sources(query, unique_urls)


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_source_quality(
    query: str,
    reference_urls: List[str],
    weight_by_position: bool = False,
) -> SourceQualityResult:
    """
    Convenience function to evaluate source quality without instantiating a scorer.
    
    Args:
        query: The original query
        reference_urls: List of reference URLs to evaluate
        weight_by_position: If True, earlier URLs get higher weight
        
    Returns:
        SourceQualityResult with overall score and detailed breakdown
        
    Example:
        >>> from jade.scoring.source_quality import evaluate_source_quality
        >>> result = evaluate_source_quality(
        ...     query="TikTok monthly active users 2024",
        ...     reference_urls=["https://newsroom.tiktok.com/...", "https://statista.com/..."]
        ... )
        >>> print(f"Quality: {result.quality_grade}, Score: {result.overall_score:.2f}")
    """
    scorer = SourceQualityScorer()
    return scorer.evaluate_sources(query, reference_urls, weight_by_position)


def get_source_tier(url: str) -> Tuple[SourceTier, str]:
    """
    Get the tier classification for a single URL.
    
    Args:
        url: URL to classify
        
    Returns:
        Tuple of (SourceTier, domain)
        
    Example:
        >>> tier, domain = get_source_tier("https://newsroom.tiktok.com/stats")
        >>> print(f"{domain}: {tier.value}")  # "newsroom.tiktok.com: T1"
    """
    scorer = SourceQualityScorer()
    evaluation = scorer._classify_source(url)
    return evaluation.tier, evaluation.domain


# =============================================================================
# Integration with Score Generator
# =============================================================================

def adjust_confidence_by_source_quality(
    base_confidence: int,
    source_quality_result: SourceQualityResult,
    quality_weight: float = 0.2,
) -> int:
    """
    Adjust verification confidence based on source quality.
    
    This can be used to modify the confidence score from verification
    based on the quality of sources used.
    
    Args:
        base_confidence: Original confidence score (0-100)
        source_quality_result: Result from source quality evaluation
        quality_weight: How much source quality affects final confidence (0.0-1.0)
        
    Returns:
        Adjusted confidence score (0-100)
        
    Example:
        >>> quality = evaluate_source_quality("TikTok users", ["https://statista.com/..."])
        >>> adjusted = adjust_confidence_by_source_quality(80, quality, 0.2)
        >>> # If quality.overall_score = 0.75, adjusted ≈ 80 * (0.8 + 0.2 * 0.75) = 76
    """
    # Source quality affects confidence: higher quality = maintain confidence
    # Lower quality = reduce confidence
    quality_factor = 1.0 - quality_weight + (quality_weight * source_quality_result.overall_score)
    
    # Apply penalty for all unknown sources
    if source_quality_result.all_unknown_sources:
        quality_factor *= 0.8  # 20% penalty
    
    adjusted = int(base_confidence * quality_factor)
    return max(0, min(100, adjusted))


# =============================================================================
# Report Generation
# =============================================================================

def generate_source_quality_report(result: SourceQualityResult) -> str:
    """
    Generate a human-readable report of source quality evaluation.
    
    Args:
        result: SourceQualityResult to report on
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "📊 SOURCE QUALITY REPORT",
        "=" * 60,
        "",
        f"Overall Score: {result.overall_score:.2f} (Grade: {result.quality_grade})",
        f"Sources Evaluated: {result.source_count}",
        "",
        "── Tier Distribution ──",
    ]
    
    tier_labels = {
        "T1": "🥇 Official/First-party",
        "T2": "🥈 Authoritative",
        "T3": "🥉 Professional",
        "T4": "⚪ Unknown",
    }
    
    for tier, count in result.tier_distribution.items():
        if count > 0:
            lines.append(f"  {tier_labels[tier]}: {count}")
    
    lines.append("")
    
    if result.recommendations:
        lines.append("── Recommendations ──")
        for rec in result.recommendations:
            lines.append(f"  • {rec}")
        lines.append("")
    
    lines.append("── Source Details ──")
    for i, src in enumerate(result.source_evaluations, 1):
        lines.append(f"  {i}. [{src.tier.value}] {src.domain}")
        lines.append(f"     Score: {src.tier_score:.2f} | URL: {src.url[:60]}...")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)