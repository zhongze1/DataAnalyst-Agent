# -*- coding: utf-8 -*-
"""生成多套「模拟真实公司业务」的测试数据集，用于测试 DataAnalyst Agent。

生成 5 个独立 Excel（每个 = 一个业务场景，各自上传为一次会话）：

    1. SaaS订阅业务_2024.xlsx    SaaS 公司：客户 / 月度订阅 / 账单流水
    2. 连锁门店销售_2024.xlsx    连锁零售：销售明细 / 门店信息 / 库存快照
    3. 广告投放数据_2024.xlsx    广告投放：投放明细 / 渠道基础
    4. 客服工单_2024.xlsx        客服中心：工单明细 / 客服人员
    5. 电商平台订单_2024.xlsx    电商平台：订单明细 / 商品主数据 / 客户主数据

设计原则（对照 app/agent/prompts.py 单表工作流与 app/tools/pandas_tools.py 工具集）：
    - 每个场景覆盖不同分析范式：趋势 / 漏斗 / 时长 / 多店对比 / RFM / 文本分类
    - 数值列供 describe_dataframe / correlation，类别列供 value_counts，日期列供趋势
    - 每个数据集刻意埋入 2~4 处业务异常，用于检验 Agent 的「风险提示」能力
    - 随机种子固定，结果完全可复现

同时产出「测试指南.md」：逐个数据集的表结构、关键指标真值、埋入异常、建议提问。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240614
OUT_DIR = Path(__file__).resolve().parent

# ================================================================ 通用工具
CITIES = ["上海", "北京", "深圳", "广州", "杭州", "成都", "武汉", "南京",
          "西安", "重庆", "苏州", "天津", "长沙", "青岛", "郑州"]
REGIONS = ["华东", "华南", "华北", "华中", "西南", "西北", "东北"]
CITY_REGION = {
    "上海": "华东", "杭州": "华东", "南京": "华东", "苏州": "华东", "青岛": "华东",
    "北京": "华北", "天津": "华北",
    "深圳": "华南", "广州": "华南",
    "武汉": "华中", "长沙": "华中", "郑州": "华中",
    "成都": "西南", "重庆": "西南",
    "西安": "西北",
}
REGION_OF = {c: CITY_REGION.get(c, "华东") for c in CITIES}

COMPANY_WORDS = ["华信", "云图", "智联", "汇通", "数聚", "恒远", "嘉泰", "弘毅",
                 "远航", "锐智", "博远", "天工", "同创", "startUp", "摩尔", "圣和",
                 "启元", "微光", "深蓝", "橡树", "星野", "青柠", "拓维", "融创"]
COMPANY_SUFFIX = ["科技有限公司", "信息技术有限公司", "网络科技有限公司",
                  "数据服务有限公司", "供应链管理有限公司", "商贸有限公司"]


def make_companies(rng: np.random.Generator, n: int) -> list[str]:
    """生成 n 个不重名的公司名。"""
    names: set[str] = set()
    while len(names) < n:
        names.add(f"{rng.choice(CITIES)}{rng.choice(COMPANY_WORDS)}"
                  f"{rng.choice(COMPANY_SUFFIX)}")
    return list(names)


def make_month_day(rng: np.random.Generator, n: int, year: int,
                   month_weights: np.ndarray | None = None) -> np.ndarray:
    """生成 n 个 2024 年内的日期（按月份权重分布）。"""
    weights = month_weights if month_weights is not None else np.ones(12)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    months = rng.choice(np.arange(1, 13), size=n, p=weights)
    days = np.array([rng.integers(1, pd.Timestamp(year, m, 1).days_in_month + 1)
                     for m in months])
    return pd.to_datetime({"year": year, "month": months, "day": days}).to_numpy()


def style_workbook(path: Path, money_cols: set[str], date_cols: set[str],
                   wide_cols: set[str]) -> None:
    """统一美化：表头样式、冻结首行、列宽、数字格式。"""
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = load_workbook(path)
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="2F5597")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = head_font
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        headers = [c.value for c in ws[1]]
        for j, name in enumerate(headers, start=1):
            width = 20 if name in wide_cols else 14
            ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=j)
                if name in money_cols:
                    cell.number_format = "#,##0.00"
                elif name in date_cols:
                    cell.number_format = "yyyy-mm-dd"
    wb.save(path)


# ================================================================ 场景 1：SaaS
def gen_saas(rng: np.random.Generator) -> dict:
    """SaaS 订阅业务：客户 / 月度订阅 / 账单流水。"""
    plans = {"基础版": 99, "专业版": 499, "企业版": 1999}
    plan_names = list(plans)
    plan_p = [0.55, 0.32, 0.13]
    industries = ["互联网", "金融", "制造业", "教育", "医疗", "零售", "物流", "政府"]

    n_cust = 420
    companies = make_companies(rng, n_cust)
    # 注册日期：全年分布式增长（越晚越多）
    reg_dates = make_month_day(rng, n_cust, 2024, np.linspace(0.6, 1.4, 12))
    plan = rng.choice(plan_names, size=n_cust, p=plan_p)
    industry = rng.choice(industries, size=n_cust,
                          p=[0.24, 0.15, 0.14, 0.12, 0.10, 0.10, 0.08, 0.07])
    city = rng.choice(CITIES, size=n_cust)
    # 约 18% 客户已流失
    churned = rng.random(n_cust) < 0.18
    status = np.where(churned, "已流失", "活跃")

    customers = pd.DataFrame({
        "客户ID": [f"CUS-{i + 1:04d}" for i in range(n_cust)],
        "客户名称": companies,
        "注册日期": reg_dates,
        "套餐": plan,
        "月费": [plans[p] for p in plan],
        "客户状态": status,
        "行业": industry,
        "城市": city,
        "地区": [REGION_OF[c] for c in city],
    })

    # ---- 月度订阅（12 月 × 3 套餐）----
    rows = []
    base = {"基础版": 150, "专业版": 90, "企业版": 35}
    for m in range(1, 13):
        grow = 1.0 + m * 0.02
        for pn in plan_names:
            begin = int(base[pn] * grow)
            new = int(rng.integers(4, 18))
            lost = int(rng.integers(2, 9))
            end = begin + new - lost
            rows.append({
                "月份": pd.Timestamp(2024, m, 1),
                "套餐": pn,
                "期初客户数": begin,
                "新增客户": new,
                "流失客户": lost,
                "期末客户数": end,
                "MRR": end * plans[pn],
            })
    monthly = pd.DataFrame(rows)

    # ---- 账单流水 ----
    n_bill = 1500
    bidx = rng.integers(0, n_cust, size=n_bill)
    b_cust = customers.iloc[bidx].reset_index(drop=True)
    bill_month = rng.integers(1, 13, size=n_bill)
    pay_status = rng.choice(["已支付", "待支付", "已逾期"], size=n_bill,
                            p=[0.86, 0.09, 0.05])
    bills = pd.DataFrame({
        "账单号": [f"BILL-2024{m:02d}-{i:04d}" for i, m in enumerate(bill_month)],
        "客户ID": b_cust["客户ID"].to_numpy(),
        "客户名称": b_cust["客户名称"].to_numpy(),
        "账单月份": pd.to_datetime({"year": 2024, "month": bill_month, "day": 1}).to_numpy(),
        "套餐": b_cust["套餐"].to_numpy(),
        "金额": b_cust["月费"].to_numpy().astype(float),
        "支付状态": pay_status,
        "支付方式": rng.choice(["对公转账", "支付宝", "微信支付", "银行卡"],
                             size=n_bill, p=[0.45, 0.25, 0.2, 0.1]),
    })

    marks: dict = {}
    # A1: 1 条退款账单（金额为负）
    neg = rng.choice(len(bills))
    bills.loc[neg, "金额"] = -499.0
    bills.loc[neg, "支付状态"] = "已退款"
    marks["退款账单"] = bills.loc[neg, "账单号"]
    # A2: 1 个重复账单号
    dup_src = rng.choice(bills.index.difference([neg]))
    dup_tgt = rng.choice(bills.index.difference([neg, dup_src]))
    bills.loc[dup_tgt, "账单号"] = bills.loc[dup_src, "账单号"]
    marks["重复账单号"] = bills.loc[dup_src, "账单号"]
    # A3: 一条「期末客户数」不等于 期初+新增-流失 的记录（数据录入错误）
    bad = rng.integers(0, len(monthly))
    monthly.loc[bad, "期末客户数"] += 7
    marks["订阅表不平"] = (
        f"{monthly.loc[bad, '月份'].date()} / {monthly.loc[bad, '套餐']}"
        "（期末客户数被改为与 期初+新增-流失 不符）"
    )

    return {
        "file": "SaaS订阅业务_2024.xlsx",
        "desc": "SaaS 订阅公司 2024 年客户、月度订阅指标与账单流水",
        "sheets": {
            "客户明细": customers,
            "月度订阅": monthly,
            "账单流水": bills,
        },
        "money": {"月费", "MRR", "金额"},
        "dates": {"注册日期", "月份", "账单月份"},
        "wide": {"客户名称"},
        "marks": marks,
    }


# ================================================================ 场景 2：连锁门店
def gen_retail(rng: np.random.Generator) -> dict:
    """连锁零售：销售明细 / 门店信息 / 库存快照。"""
    n_store = 20
    stores = []
    for i in range(n_store):
        city = rng.choice(CITIES)
        stype = rng.choice(["旗舰店", "标准店", "社区店"], p=[0.2, 0.5, 0.3])
        stores.append({
            "门店ID": f"ST{i + 1:03d}",
            "门店名称": f"{city}{stype}{i + 1:02d}",
            "城市": city,
            "区域": REGION_OF[city],
            "门店类型": stype,
            "开业日期": pd.Timestamp(rng.integers(2018, 2024),
                                    rng.integers(1, 13), rng.integers(1, 28)),
            "营业面积": int(rng.integers(120, 1200)),
            "店长": f"{rng.choice(list('赵钱孙李周吴郑王'))}{rng.choice(['伟','芳','强','静','磊','敏','军','丽'])}",
        })
    store_df = pd.DataFrame(stores)

    # 商品维度
    goods = []
    cats = {"食品饮料": (3, 45), "日用百货": (5, 80), "家用电器": (200, 3000),
            "服饰鞋帽": (30, 500), "美妆个护": (20, 350)}
    for i, (cat, (lo, hi)) in enumerate([(c, v) for c, v in cats.items()]):
        for k in range(6):
            price = round(float(rng.uniform(lo, hi)), 1)
            goods.append({"SKU": f"SKU{len(goods) + 1:04d}", "商品名称": f"{cat}商品{k + 1:02d}",
                          "品类": cat, "单价": price})
    goods_df = pd.DataFrame(goods)

    # 销售明细（约 3200 行）
    n = 3200
    sidx = rng.integers(0, n_store, size=n)
    gidx = rng.integers(0, len(goods_df), size=n)
    s_store = store_df.iloc[sidx].reset_index(drop=True)
    s_goods = goods_df.iloc[gidx].reset_index(drop=True)
    # 日期从门店开业后才可能销售；统一用 2024 年
    dates = make_month_day(rng, n, 2024, np.array([1.0, 0.9, 1.0, 1.05, 1.1, 1.25,
                                                   1.0, 0.95, 1.05, 1.15, 1.4, 1.2]))
    qty = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 10], size=n)
    disc = np.clip(rng.normal(0.95, 0.06, n), 0.6, 1.0).round(2)
    member = rng.choice(["是", "否"], size=n, p=[0.6, 0.4])

    sales = pd.DataFrame({
        "交易号": [f"TX2024{i + 1:05d}" for i in range(n)],
        "日期": dates,
        "门店ID": s_store["门店ID"].to_numpy(),
        "门店名称": s_store["门店名称"].to_numpy(),
        "城市": s_store["城市"].to_numpy(),
        "区域": s_store["区域"].to_numpy(),
        "SKU": s_goods["SKU"].to_numpy(),
        "商品名称": s_goods["商品名称"].to_numpy(),
        "品类": s_goods["品类"].to_numpy(),
        "销售数量": qty.astype(int),
        "单价": s_goods["单价"].to_numpy(),
        "折扣率": disc,
        "会员": member,
    })
    sales["销售额"] = (sales["销售数量"] * sales["单价"] * sales["折扣率"]).round(2)

    # 库存快照（月末 × 门店 × 部分 SKU）
    snap_dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    inv_rows = []
    for dt in snap_dates:
        for s in range(n_store):
            for g in rng.choice(len(goods_df), size=6, replace=False):
                inv_rows.append({
                    "快照日期": dt,
                    "门店ID": store_df.loc[s, "门店ID"],
                    "SKU": goods_df.loc[g, "SKU"],
                    "库存量": int(rng.integers(0, 400)),
                    "安全库存": int(rng.integers(30, 120)),
                    "周转天数": round(float(rng.gamma(2, 12)), 1),
                })
    inventory = pd.DataFrame(inv_rows)

    marks: dict = {}
    # A1: 1 笔负销售额（集中退货录入）
    neg = rng.choice(len(sales))
    sales.loc[neg, "销售数量"] = -1
    sales.loc[neg, "销售额"] = -(sales.loc[neg, "单价"] * sales.loc[neg, "折扣率"]).round(2)
    marks["负销售额"] = sales.loc[neg, "交易号"]
    # A2: 销售日期早于门店开业日期（数据错位）
    bad_store = store_df.index[0]
    later = sales.index[(sales["门店ID"] == store_df.loc[bad_store, "门店ID"])][0]
    sales.loc[later, "日期"] = pd.Timestamp("2017-06-01")
    marks["日期早于开业"] = {
        "交易号": sales.loc[later, "交易号"],
        "日期": "2017-06-01",
        "门店": store_df.loc[bad_store, "门店名称"],
        "开业日期": str(store_df.loc[bad_store, "开业日期"].date()),
    }
    # A3: 库存周转天数极端值（录入为 999）
    ext = rng.choice(len(inventory))
    inventory.loc[ext, "周转天数"] = 999.0
    marks["极端周转天数"] = f"{inventory.loc[ext, '门店ID']} / {inventory.loc[ext, 'SKU']} = 999 天"

    return {
        "file": "连锁门店销售_2024.xlsx",
        "desc": "20 家连锁门店 2024 年销售明细、门店信息与月末库存快照",
        "sheets": {
            "销售明细": sales,
            "门店信息": store_df,
            "库存快照": inventory,
        },
        "money": {"单价", "销售额"},
        "dates": {"日期", "开业日期", "快照日期"},
        "wide": {"门店名称", "商品名称"},
        "marks": marks,
    }# ================================================================ 场景 3：广告投放
def gen_ads(rng: np.random.Generator) -> dict:
    """广告投放：投放明细（日 × 渠道 × 广告组）+ 渠道基础表。"""
    channels = ["巨量引擎", "腾讯广告", "百度广告", "小红书", "知乎", "B站"]
    ch_cpc = {"巨量引擎": 1.8, "腾讯广告": 1.5, "百度广告": 2.2,
              "小红书": 2.6, "知乎": 1.9, "B站": 1.2}
    ch_ctr = {"巨量引擎": 0.035, "腾讯广告": 0.028, "百度广告": 0.022,
              "小红书": 0.048, "知乎": 0.030, "B站": 0.052}
    ch_cvr = {"巨量引擎": 0.021, "腾讯广告": 0.025, "百度广告": 0.030,
              "小红书": 0.018, "知乎": 0.022, "B站": 0.012}
    campaigns = {
        "巨量引擎": ["暑期大促", "新客拉新", "品牌曝光"],
        "腾讯广告": ["朋友圈种草", "新客拉新", "节日促销"],
        "百度广告": ["搜索品牌词", "搜索品类词", "信息流"],
        "小红书": ["达人种草", "笔记推广", "新客拉新"],
        "知乎": ["问答植入", "专栏推广"],
        "B站": ["UP主合作", "开屏广告"],
    }

    rows = []
    for ch in channels:
        for camp in campaigns[ch]:
            # 每个广告组在 2024 年随机活跃一段时间
            start = rng.integers(1, 10)
            span = int(rng.integers(30, 120))
            dates = pd.date_range(f"2024-{start:02d}-01", periods=span, freq="D")
            for dt in dates:
                imp = int(rng.integers(20000, 300000))
                clicks = int(imp * ch_ctr[ch] * rng.normal(1.0, 0.08))
                if clicks < 1:
                    clicks = 1
                conv = int(clicks * ch_cvr[ch] * rng.normal(1.0, 0.1))
                if conv < 0:
                    conv = 0
                cost = round(clicks * ch_cpc[ch] * rng.normal(1.0, 0.06), 2)
                # 客单价与渠道相关
                aov = round(float(rng.normal(320, 60)), 2)
                rows.append({
                    "日期": dt,
                    "渠道": ch,
                    "广告组": camp,
                    "曝光量": imp,
                    "点击量": clicks,
                    "转化数": conv,
                    "消耗": cost,
                    "成交金额": round(conv * max(aov, 50), 2),
                })
    ads = pd.DataFrame(rows)

    ch_base = pd.DataFrame([{
        "渠道": ch,
        "开户时间": pd.Timestamp(2022, rng.integers(1, 13), 1),
        "结算方式": rng.choice(["预付", "月结"]),
        "负责人": f"{rng.choice(['张','王','李','刘','陈'])}"
                  f"{rng.choice(['明','华','超','静','洋'])}",
        "返点比例": round(float(rng.uniform(0.02, 0.12)), 3),
    } for ch in channels])

    marks: dict = {}
    # A1: 3 条点击量 > 曝光量（埋点异常，业务上不可能）
    picks = rng.choice(ads.index, size=3, replace=False)
    for p in picks:
        ads.loc[p, "点击量"] = ads.loc[p, "曝光量"] + int(rng.integers(500, 3000))
    marks["点击大于曝光"] = [f"{ads.loc[p, '日期'].date()} {ads.loc[p, '渠道']}"
                             for p in picks]
    # A2: 1 条消耗为 0 但有转化（免费流量混入或漏记）
    zero = rng.choice(ads.index.difference(picks))
    ads.loc[zero, "消耗"] = 0.0
    marks["零消耗有转化"] = f"{ads.loc[zero, '日期'].date()} {ads.loc[zero, '渠道']}"
    # A3: 缺失值
    ads.loc[rng.choice(ads.index, size=40, replace=False), "成交金额"] = np.nan
    ads.loc[rng.choice(ads.index, size=15, replace=False), "广告组"] = np.nan
    return {
        "file": "广告投放数据_2024.xlsx",
        "desc": "6 个投放渠道、多个广告组的 2024 年逐日投放明细",
        "sheets": {"投放明细": ads, "渠道基础": ch_base},
        "money": {"消耗", "成交金额"},
        "dates": {"日期", "开户时间"},
        "wide": {"广告组"},
        "marks": marks,
    }


# ================================================================ 场景 4：客服工单
def gen_service(rng: np.random.Generator) -> dict:
    """客服中心：工单明细 + 客服人员。"""
    types = ["咨询", "投诉", "退换货", "技术支持", "账单问题"]
    priorities = ["低", "中", "高", "紧急"]
    channels = ["电话", "在线客服", "邮件", "工单系统", "微信"]
    status_pool = ["已解决", "处理中", "已关闭", "待跟进"]

    agents = []
    surnames = list("赵钱孙李周吴郑王冯陈")
    givens = ["敏", "静", "丽", "强", "磊", "洋", "艳", "杰", "娟", "涛"]
    for i in range(14):
        agents.append({
            "客服ID": f"AG{i + 1:03d}",
            "客服姓名": f"{rng.choice(surnames)}{rng.choice(givens)}",
            "所属组": rng.choice(["售前组", "售后组", "技术组"]),
            "入职日期": pd.Timestamp(rng.integers(2020, 2024), rng.integers(1, 13), 1),
            "职级": rng.choice(["初级", "中级", "高级"], p=[0.4, 0.4, 0.2]),
        })
    agent_df = pd.DataFrame(agents)

    n = 2600
    created = make_month_day(rng, n, 2024, np.array([1.1, 0.8, 1.0, 1.0, 1.05, 1.2,
                                                     1.0, 0.95, 1.05, 1.1, 1.35, 1.15]))
    # 响应时长（分钟）：对数正态，投诉/技术类更长
    ttype = rng.choice(types, size=n, p=[0.34, 0.16, 0.2, 0.18, 0.12])
    base_min = np.where(np.isin(ttype, ["投诉", "技术支持"]), 25, 10)
    response_min = np.clip(rng.lognormal(np.log(base_min), 0.7, n), 1, 1440).round(1)
    # 解决时长（小时）：必须 >= 响应时长
    resolve_h = (response_min / 60.0) + np.clip(rng.gamma(2.5, 3.0, n), 0.2, 200)
    resolve_h = resolve_h.round(2)

    status = rng.choice(status_pool, size=n, p=[0.7, 0.12, 0.12, 0.06])
    resolved_flag = np.isin(status, ["已解决", "已关闭"])
    # 满意度只可能来自「已解决/已关闭」工单，且约 65% 工单会回填评分
    satisfaction = np.full(n, np.nan)
    resolved_idx = np.where(resolved_flag)[0]
    filled = rng.choice(resolved_idx, size=int(len(resolved_idx) * 0.65), replace=False)
    satisfaction[filled] = rng.choice([1, 2, 3, 4, 5], size=len(filled),
                                      p=[0.04, 0.06, 0.15, 0.38, 0.37])

    ai = rng.integers(0, len(agents), size=n)
    service = pd.DataFrame({
        "工单号": [f"TK2024{i + 1:05d}" for i in range(n)],
        "创建时间": created,
        "客户ID": [f"C{rng.integers(1, 900):04d}" for _ in range(n)],
        "工单类型": ttype,
        "优先级": rng.choice(priorities, size=n, p=[0.3, 0.4, 0.22, 0.08]),
        "来源渠道": rng.choice(channels, size=n, p=[0.3, 0.28, 0.12, 0.18, 0.12]),
        "客服ID": agent_df["客服ID"].to_numpy()[ai],
        "客服姓名": agent_df["客服姓名"].to_numpy()[ai],
        "所属组": agent_df["所属组"].to_numpy()[ai],
        "工单状态": status,
        "响应时长_分钟": response_min,
        "解决时长_小时": resolve_h,
        "满意度评分": satisfaction,
        "是否升级": rng.choice(["是", "否"], size=n, p=[0.12, 0.88]),
        "客户留言": [rng.choice([
            "问题已解决，感谢", "希望尽快处理", "服务态度很好", "响应有点慢",
            "再次反馈同样问题", "需要电话回访", "非常不满意", "已经解决了"])
            if rng.random() < 0.35 else np.nan for _ in range(n)],
    })

    marks: dict = {}
    # A1: 未解决(处理中/待跟进)的工单却填了满意度评分——逻辑矛盾
    unresolved = service.index[~resolved_flag]
    u = rng.choice(unresolved, size=5, replace=False)
    service.loc[u[:4], "满意度评分"] = rng.integers(3, 6, size=4)
    service.loc[u[4], "满意度评分"] = 6  # 越界评分（应只在 1~5）
    marks["未解决却有评分"] = service.loc[u, "工单号"].tolist()
    marks["越界评分(6)"] = service.loc[u[4], "工单号"]
    # A2: 3 条解决时长 < 响应时长（时序错误）
    resolve = rng.choice(service.index, size=3, replace=False)
    service.loc[resolve, "解决时长_小时"] = (
        service.loc[resolve, "响应时长_分钟"] / 60.0 * 0.5
    ).round(2)
    marks["解决早于响应"] = service.loc[resolve, "工单号"].tolist()
    # A3: 2 条「响应时长 = 0」——秒级响应不可能，疑似漏记
    zero_resp = rng.choice(service.index.difference(resolve).difference(u), size=2,
                           replace=False)
    service.loc[zero_resp, "响应时长_分钟"] = 0.0
    marks["零响应时长"] = service.loc[zero_resp, "工单号"].tolist()
    return {
        "file": "客服工单_2024.xlsx",
        "desc": "14 名客服、2600 条 2024 年工单的处理时长、状态与满意度",
        "sheets": {"工单明细": service, "客服人员": agent_df},
        "money": set(),
        "dates": {"创建时间", "入职日期"},
        "wide": {"客服姓名", "客户留言"},
        "marks": marks,
    }


# ================================================================ 场景 5：电商平台
def gen_ecommerce(rng: np.random.Generator) -> dict:
    """电商平台：订单明细 + 商品主数据 + 客户主数据（支持 RFM 分析）。"""
    cats = {
        "手机数码": (500, 8000), "家用电器": (300, 6000), "服饰鞋包": (80, 900),
        "美妆护肤": (50, 800), "食品生鲜": (20, 300), "家居家装": (100, 3000),
        "母婴玩具": (40, 700), "运动户外": (120, 1500),
    }
    products = []
    for cat, (lo, hi) in cats.items():
        for k in range(4):
            price = round(float(rng.uniform(lo, hi)), 2)
            products.append({
                "商品ID": f"P{len(products) + 1:04d}",
                "商品名称": f"{cat}{['A', 'B', 'C', 'D'][k]}款",
                "品类": cat,
                "标准售价": price,
                "成本价": round(price * rng.uniform(0.5, 0.75), 2),
                "供应商": f"SUP{rng.integers(1, 20):03d}",
                "上架日期": pd.Timestamp(rng.integers(2021, 2024), rng.integers(1, 13), 1),
            })
    prod_df = pd.DataFrame(products)

    n_cust = 700
    customer_names = []
    surnames = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张")
    givens = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋", "勇",
              "艳", "杰", "娟", "涛", "明", "超", "霞", "平", "刚"]
    for _ in range(n_cust):
        customer_names.append(f"{rng.choice(surnames)}{rng.choice(givens)}")
    reg = make_month_day(rng, n_cust, 2024, np.linspace(1.6, 0.6, 12))
    level = rng.choice(["普通", "银牌", "金牌", "钻石"], size=n_cust,
                       p=[0.5, 0.28, 0.16, 0.06])
    cust_city = rng.choice(CITIES, size=n_cust)
    customers = pd.DataFrame({
        "客户ID": [f"C{i + 1:04d}" for i in range(n_cust)],
        "客户姓名": customer_names,
        "注册日期": reg,
        "会员等级": level,
        "城市": cust_city,
        "地区": [REGION_OF[c] for c in cust_city],
        "手机号": [f"1{rng.integers(3, 9)}{rng.integers(10**8, 10**9 - 1)}"
                 for _ in range(n_cust)],
    })

    # 订单明细：客户复购（头部客户更多单，且高等级会员下单更活跃）
    n_order = 3600
    cp = rng.dirichlet(np.ones(n_cust) * 0.35)
    level_boost = customers["会员等级"].map(
        {"普通": 0.7, "银牌": 1.0, "金牌": 1.6, "钻石": 2.4}).to_numpy()
    cp = cp * level_boost
    cp = cp / cp.sum()
    cidx = rng.choice(n_cust, size=n_order, p=cp)
    o_cust = customers.iloc[cidx].reset_index(drop=True)
    pidx = rng.integers(0, len(prod_df), size=n_order)
    o_prod = prod_df.iloc[pidx].reset_index(drop=True)
    odates = make_month_day(rng, n_order, 2024,
                            np.array([1.0, 0.85, 1.0, 1.0, 1.1, 1.5, 0.95, 0.9,
                                      1.05, 1.15, 1.9, 1.3]))
    qty = rng.choice([1, 1, 1, 1, 2, 2, 3, 5], size=n_order)
    disc = np.clip(rng.normal(0.93, 0.07, n_order), 0.5, 1.0).round(2)

    orders = pd.DataFrame({
        "订单号": [f"ORD2024{i + 1:06d}" for i in range(n_order)],
        "下单日期": odates,
        "客户ID": o_cust["客户ID"].to_numpy(),
        "客户姓名": o_cust["客户姓名"].to_numpy(),
        "会员等级": o_cust["会员等级"].to_numpy(),
        "地区": o_cust["地区"].to_numpy(),
        "商品ID": o_prod["商品ID"].to_numpy(),
        "商品名称": o_prod["商品名称"].to_numpy(),
        "品类": o_prod["品类"].to_numpy(),
        "销售数量": qty.astype(int),
        "单价": o_prod["标准售价"].to_numpy(),
        "折扣率": disc,
        "支付方式": rng.choice(["支付宝", "微信支付", "银行卡", "花呗", "货到付款"],
                            size=n_order, p=[0.34, 0.3, 0.14, 0.17, 0.05]),
        "订单状态": rng.choice(["已完成", "已发货", "待发货", "已取消", "退款中"],
                            size=n_order, p=[0.72, 0.13, 0.07, 0.05, 0.03]),
        "物流方式": rng.choice(["顺丰", "中通", "圆通", "京东物流", "邮政"],
                            size=n_order, p=[0.3, 0.25, 0.18, 0.2, 0.07]),
    })
    orders["订单金额"] = (orders["销售数量"] * orders["单价"] * orders["折扣率"]).round(2)
    orders["成本"] = (orders["销售数量"] * o_prod["成本价"].to_numpy()).round(2)
    orders["毛利"] = (orders["订单金额"] - orders["成本"]).round(2)

    marks: dict = {}
    # A1: 2 笔超大批量订单（企业团购 / 疑似刷单）
    big = rng.choice(len(orders), size=2, replace=False)
    orders.loc[big, "销售数量"] = 800
    orders.loc[big, "订单金额"] = (
        orders.loc[big, "销售数量"] * orders.loc[big, "单价"]
        * orders.loc[big, "折扣率"]).round(2)
    orders.loc[big, "毛利"] = (orders.loc[big, "订单金额"] - orders.loc[big, "成本"]).round(2)
    marks["超大批量订单"] = orders.loc[big, "订单号"].tolist()
    # A2: 1 个重复订单号
    d1, d2 = rng.choice(orders.index.difference(big), size=2, replace=False)
    orders.loc[d2, "订单号"] = orders.loc[d1, "订单号"]
    marks["重复订单号"] = orders.loc[d1, "订单号"]
    # A3: 已取消订单却有金额（金额应为 0）
    cancel = orders.index[orders["订单状态"] == "已取消"]
    if len(cancel):
        cc = rng.choice(cancel, size=min(6, len(cancel)), replace=False)
        marks["已取消仍有金额"] = orders.loc[cc, "订单号"].tolist()
    # A4: 缺失值
    orders.loc[rng.choice(orders.index, size=int(len(orders) * 0.02), replace=False),
               "物流方式"] = np.nan
    customers.loc[rng.choice(customers.index, size=8, replace=False),
                  "手机号"] = np.nan
    return {
        "file": "电商平台订单_2024.xlsx",
        "desc": "电商平台 700 客户、3600 笔订单及商品/客户主数据（含 RFM 分析所需字段）",
        "sheets": {"订单明细": orders, "商品主数据": prod_df, "客户主数据": customers},
        "money": {"单价", "订单金额", "成本", "毛利", "标准售价", "成本价"},
        "dates": {"下单日期", "上架日期", "注册日期"},
        "wide": {"商品名称", "客户姓名"},
        "marks": marks,
    }


# ================================================================ 主流程
GENERATORS = [gen_saas, gen_retail, gen_ads, gen_service, gen_ecommerce]


def main() -> None:
    rng = np.random.default_rng(SEED)
    built = []
    for gen in GENERATORS:
        data = gen(rng)
        path = OUT_DIR / data["file"]
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            first = True
            for sheet_name, df in data["sheets"].items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                if first:
                    first = False
        style_workbook(path, data["money"], data["dates"], data["wide"])
        built.append(data)
        total = sum(len(df) for df in data["sheets"].values())
        print(f"[OK] {data['file']:28s} "
              f"{len(data['sheets'])} 表 / {total} 行 / {path.stat().st_size / 1024:.1f} KB")

    # 打印所有埋入异常，便于核对
    print("\n--- 埋入异常锚点 ---")
    for data in built:
        print(f"\n{data['file']}")
        for k, v in data["marks"].items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()