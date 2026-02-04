"""
Multilabel Label Utilities.

Helper functions for extracting and inferring multi-label classifications.
"""

from typing import Dict, List, Any


def extract_labels_from_item(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract multi-label classification from a data item.
    
    This helper function standardizes the label extraction from dataset items
    that may have different field naming conventions.
    
    Args:
        item_data: Dictionary containing the data item with label fields.
    
    Returns:
        Standardized labels dictionary with keys:
        - L1_primary_intent
        - L2_information_need
        - L3_constraints
    """
    return {
        "L1_primary_intent": item_data.get("L1_primary_intent", ""),
        "L2_information_need": item_data.get("L2_information_need", []),
        "L3_constraints": item_data.get("L3_constraints", []),
    }


def infer_labels_from_query(query: str) -> Dict[str, Any]:
    """
    Automatically infer multi-label classification from query text.
    
    This function uses keyword-based heuristics to classify queries
    when no pre-existing labels are available.
    
    Args:
        query: The query text to analyze.
    
    Returns:
        Inferred labels dictionary with keys:
        - L1_primary_intent
        - L2_information_need
        - L3_constraints
    """
    import re
    q = query.lower()
    
    # Initialize result
    l1_intent = ""
    l2_needs: List[str] = []
    l3_constraints: List[str] = []
    
    # ========== L1 Intent Detection (order matters - more specific first) ==========
    # Product development / OEM patterns (check first)
    if re.search(r"\b(oem|odm|开模|打样|私模)\b", q) or (
        re.search(r"定制|customize", q) and re.search(r"工厂|factory|厂家", q)
    ):
        l1_intent = "product_development"
    # Supplier sourcing patterns
    elif re.search(r"supplier|sourcing|供应商|货源|找.*厂|厂家", q) or (
        re.search(r"factory|manufacturer|工厂", q) and 
        re.search(r"找|find|推荐|recommend|哪些|which", q)
    ):
        l1_intent = "supplier_sourcing"
    # Market research patterns
    elif re.search(r"\b(market|市场|竞争格局|行业)\b", q) or (
        re.search(r"分析|analysis|研究|research", q) and 
        re.search(r"趋势|trend|竞品|competitor", q)
    ):
        l1_intent = "market_research"
    # Default to product discovery (finding products to sell)
    else:
        l1_intent = "product_discovery"
    
    # ========== L2 Information Need Detection ==========
    # Trending analysis
    if re.search(r"趋势|trend|热度|流行|popular|viral|火|爆", q):
        l2_needs.append("trending_analysis")
    
    # Sales data
    if re.search(r"销量|销售|sales|revenue|bsr|best[- ]?sell|gmv|热销|卖得好", q):
        l2_needs.append("sales_data")
    
    # Review analysis
    if re.search(r"评价|评论|review|rating|feedback|口碑|星级", q):
        l2_needs.append("review_analysis")
    
    # Competitor analysis
    if re.search(r"竞品|竞争|competitor|competition|对手|竞争格局|对比.*品牌", q):
        l2_needs.append("competitor_analysis")
    
    # Price comparison
    if re.search(r"价格|报价|price|pricing|cost|成本|单价|多少钱", q):
        l2_needs.append("price_comparison")
    
    # Platform data
    if re.search(r"amazon|亚马逊|alibaba|阿里|1688|tiktok|抖音|youtube|shopee|lazada|ebay|etsy|淘宝|天猫|拼多多", q):
        l2_needs.append("platform_data")
    
    # Supplier evaluation
    if re.search(r"评估|考察|evaluate|assess|对比.*供应商|compare.*supplier|哪家.*好", q) or (
        l1_intent == "supplier_sourcing" and re.search(r"对比|比较|推荐", q)
    ):
        l2_needs.append("supplier_evaluation")
    
    # ========== L3 Constraint Detection ==========
    # Certification required
    if re.search(r"认证|certif|fda|ce[^a-z]|en71|rohs|reach|fcc|ul[^a-z]|prop[- ]?65|lfgb|bsci|合规|标准", q):
        l3_constraints.append("certification_required")
    
    # MOQ / Price constraint
    if re.search(r"moq|起订|最低.*量|budget|预算|低于|不超过|under \$|less than \$|\d+.*以内", q):
        l3_constraints.append("moq_price_constraint")
    
    # Logistics / Shipping
    if re.search(r"物流|运费|发货|交期|shipping|deliver|logistics|ddp|fob|cif|freight|lead time|到.*国", q):
        l3_constraints.append("logistics_shipping")
    
    # Customization / OEM
    if re.search(r"定制|oem|odm|logo|贴牌|打标|私模|brand|private label", q):
        l3_constraints.append("customization_oem")
    
    # Region specific
    if re.search(r"美国|usa|us market|欧洲|europe|uk|英国|germany|德国|japan|日本|australia|澳洲|东南亚", q):
        l3_constraints.append("region_specific")
    
    # Quality specification
    if re.search(r"质量|材质|规格|参数|quality|material|spec|specification|要求.*是", q):
        l3_constraints.append("quality_specification")
    
    return {
        "L1_primary_intent": l1_intent,
        "L2_information_need": l2_needs,
        "L3_constraints": l3_constraints,
    }
