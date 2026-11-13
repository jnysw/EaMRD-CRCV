"""Authoritative domains for rumor / fact-check evidence search."""

from __future__ import annotations

# 国家级权威辟谣与主流媒体
RUMOR_PLATFORMS = [
    "piyao.org.cn",       # 中国互联网联合辟谣平台
    "news.cn",            # 新华网
    "people.com.cn",      # 人民网
    "news.cctv.com",      # 央视新闻
    "chinanews.com.cn",   # 中国新闻网
]

# 专业事实核查
FACT_CHECK_PLATFORMS = [
    "factpaper.cn",       # 澎湃新闻·明查
    "fact.qq.com",        # 腾讯较真
]

# 政府部门官方信息源
GOV_PLATFORMS = [
    "nhc.gov.cn",         # 国家卫健委
    "mps.gov.cn",         # 公安部
    "mem.gov.cn",         # 应急管理部
    "cma.gov.cn",         # 中国气象局
    "ceic.ac.cn",         # 中国地震台网中心
    "ndcpa.gov.cn",       # 国家疾控局
]

# 教育与行业辟谣
OTHER_AUTHORITY = [
    "jyb.cn",             # 全国教育辟谣平台（jypy.jyb.cn）
    "kepuchina.cn",       # 科普中国
    "12377.cn",           # 中国互联网举报中心
]

# 全部权威域名（博查 include 最多 100 个，当前 16 个）
ALL_AUTHORITY_DOMAINS = (
    RUMOR_PLATFORMS + FACT_CHECK_PLATFORMS + GOV_PLATFORMS + OTHER_AUTHORITY
)

# 分层搜索：先辟谣/核查平台，再政府部门，最后全量权威域
SEARCH_TIERS: dict[str, list[str]] = {
    "rumor_factcheck": RUMOR_PLATFORMS + FACT_CHECK_PLATFORMS,
    "gov": GOV_PLATFORMS,
    "all_authority": ALL_AUTHORITY_DOMAINS,
}


def domains_to_include(domains: list[str]) -> str:
    """Convert domain list to Bocha API include string (| separated)."""
    return "|".join(d.strip() for d in domains if d.strip())


def get_tier_include(tier: str) -> str | None:
    domains = SEARCH_TIERS.get(tier)
    if not domains:
        return None
    return domains_to_include(domains)
