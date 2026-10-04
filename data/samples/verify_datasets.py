# -*- coding: utf-8 -*-
"""校验 5 份业务测试数据集：断言每个埋入异常真实存在，并打印关键指标真值。

用途：
    - 生成数据后自检（确保「测试指南.md」宣称的异常确实在数据里）
    - 核对 DataAnalyst Agent 的回答是否算对

所有读取都走项目自己的 ``SessionStore``（与上传后 Agent 读盘同一路径）。

运行:  python verify_datasets.py
退出码: 0 = 全部断言通过；1 = 有断言失败（会打印失败项）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.storage.session_store import SessionStore  # noqa: E402

SAMPLES = Path(__file__).resolve().parent
rd = SessionStore.read_frame

_checks: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _checks.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def head(title: str) -> None:
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


# ---------------------------------------------------------------- 1. SaaS
def verify_saas() -> None:
    head("SaaS订阅业务_2024.xlsx")
    p = SAMPLES / "SaaS订阅业务_2024.xlsx"
    cus, mon, bil = rd(p, "客户明细"), rd(p, "月度订阅"), rd(p, "账单流水")

    print(f"客户 {len(cus)} | 月度订阅 {len(mon)} | 账单 {len(bil)}")
    check("工作表结构", len(cus) == 420 and len(mon) == 36 and len(bil) == 1500)
    check("12月MRR合计=174446", abs(mon[mon['月份'] == '2024-12-01']['MRR'].sum() - 174446) < 1,
          f"实算 {mon[mon['月份'] == '2024-12-01']['MRR'].sum():.0f}")
    check("账单总额=665102", abs(bil["金额"].sum() - 665102) < 1,
          f"实算 {bil['金额'].sum():.0f}")

    neg = bil[bil["金额"] < 0]
    check("退款账单(金额<0)存在", len(neg) == 1, f"{neg['账单号'].tolist()}")
    dup = bil["账单号"][bil["账单号"].duplicated(keep=False)]
    check("重复账单号存在", len(set(dup)) == 1, f"{sorted(set(dup))}")
    mon["_calc"] = mon["期初客户数"] + mon["新增客户"] - mon["流失客户"]
    bad = mon[mon["期末客户数"] != mon["_calc"]]
    check("订阅表存在不平记录", len(bad) == 1,
          f"{bad['月份'].dt.date.tolist()} {bad['套餐'].tolist()}")


# ---------------------------------------------------------------- 2. 连锁门店
def verify_retail() -> None:
    head("连锁门店销售_2024.xlsx")
    p = SAMPLES / "连锁门店销售_2024.xlsx"
    s, st, inv = rd(p, "销售明细"), rd(p, "门店信息"), rd(p, "库存快照")

    print(f"销售 {len(s)} | 门店 {len(st)} | 库存 {len(inv)}")
    check("工作表结构", len(s) == 3200 and len(st) == 20 and len(inv) == 1440)
    check("总销售额=4440228.47", abs(s["销售额"].sum() - 4440228.47) < 0.5,
          f"实算 {s['销售额'].sum():.2f}")

    check("负销售额存在", (s["销售额"] < 0).sum() == 1,
          f"{s[s['销售额'] < 0]['交易号'].tolist()}")
    early = s[s["日期"] < "2024-01-01"]
    check("存在早于2024的销售日期", len(early) == 1,
          f"{early['交易号'].tolist()} @ {early['日期'].dt.date.tolist()}")
    # 该日期确实早于对应门店的开业日期
    if len(early):
        tr = early.iloc[0]
        open_date = st[st["门店ID"] == tr["门店ID"]]["开业日期"].iloc[0]
        check("该日期早于门店开业日期", tr["日期"] < open_date,
              f"销售 {tr['日期'].date()} < 开业 {open_date.date()}")
    check("极端周转天数(999)存在", (inv["周转天数"] == 999).sum() == 1)
    low = (inv["库存量"] < inv["安全库存"]).sum()
    check("库存<安全库存记录数=228", low == 228, f"实算 {low}")


# ---------------------------------------------------------------- 3. 广告投放
def verify_ads() -> None:
    head("广告投放数据_2024.xlsx")
    p = SAMPLES / "广告投放数据_2024.xlsx"
    a, ch = rd(p, "投放明细"), rd(p, "渠道基础")

    print(f"投放明细 {len(a)} | 渠道 {len(ch)}")
    check("工作表结构", len(a) == 1218 and len(ch) == 6)
    check("总消耗=12601565.78", abs(a["消耗"].sum() - 12601565.78) < 1,
          f"实算 {a['消耗'].sum():.2f}")
    roas = a["成交金额"].sum() / a["消耗"].sum()
    check("整体ROAS≈3.33", abs(roas - 3.33) < 0.01, f"实算 {roas:.2f}")

    bad = a[a["点击量"] > a["曝光量"]]
    check("点击>曝光 记录数=3", len(bad) == 3,
          bad["日期"].dt.date.tolist())
    zero = a[(a["消耗"] == 0) & (a["转化数"] > 0)]
    check("零消耗有转化 记录数=1", len(zero) == 1, zero["日期"].dt.date.tolist())
    check("广告组缺失=15", int(a["广告组"].isna().sum()) == 15)
    check("成交金额缺失=40", int(a["成交金额"].isna().sum()) == 40)


# ---------------------------------------------------------------- 4. 客服工单
def verify_service() -> None:
    head("客服工单_2024.xlsx")
    p = SAMPLES / "客服工单_2024.xlsx"
    t, ag = rd(p, "工单明细"), rd(p, "客服人员")

    print(f"工单 {len(t)} | 客服 {len(ag)}")
    check("工作表结构", len(t) == 2600 and len(ag) == 14)
    check("平均响应时长≈19.1", abs(t["响应时长_分钟"].mean() - 19.1) < 0.1,
          f"实算 {t['响应时长_分钟'].mean():.1f}")
    check("平均解决时长≈7.83", abs(t["解决时长_小时"].mean() - 7.83) < 0.05,
          f"实算 {t['解决时长_小时'].mean():.2f}")

    resolved = t["工单状态"].isin(["已解决", "已关闭"])
    unres_scored = t[~resolved & t["满意度评分"].notna()]
    check("未解决却有评分=5", len(unres_scored) == 5,
          unres_scored["工单号"].tolist())
    check("越界评分(>5)存在", (t["满意度评分"] > 5).sum() == 1,
          f"{t[t['满意度评分'] > 5]['工单号'].tolist()}")
    bad_time = t[t["解决时长_小时"] < t["响应时长_分钟"] / 60]
    check("解决时长<响应时长=3", len(bad_time) == 3, bad_time["工单号"].tolist())
    check("零响应时长=2", (t["响应时长_分钟"] == 0).sum() == 2,
          f"{t[t['响应时长_分钟'] == 0]['工单号'].tolist()}")
    # 语义健康：未解决工单不应有评分（排除刻意埋入的 5 条后）
    natural = t[~resolved & t["满意度评分"].notna()]
    check("未解决评分仅为刻意埋入(非自然产生)", len(natural) == 5)


# ---------------------------------------------------------------- 5. 电商平台
def verify_ecommerce() -> None:
    head("电商平台订单_2024.xlsx")
    p = SAMPLES / "电商平台订单_2024.xlsx"
    o, pr, cu = rd(p, "订单明细"), rd(p, "商品主数据"), rd(p, "客户主数据")

    print(f"订单 {len(o)} | 商品 {len(pr)} | 客户 {len(cu)}")
    check("工作表结构", len(o) == 3600 and len(pr) == 32 and len(cu) == 700)
    valid = o[o["订单状态"] != "已取消"]
    check("剔除已取消后金额≈9252156.62",
          abs(valid["订单金额"].sum() - 9252156.62) < 1,
          f"实算 {valid['订单金额'].sum():.2f}")
    gm = valid["毛利"].sum() / valid["订单金额"].sum() * 100
    check("毛利率≈37.36%", abs(gm - 37.36) < 0.1, f"实算 {gm:.2f}%")

    big = o[o["销售数量"] > 100]
    check("超大批量订单=2", len(big) == 2, big["订单号"].tolist())
    dup = o["订单号"][o["订单号"].duplicated(keep=False)]
    check("重复订单号=1", len(set(dup)) == 1, sorted(set(dup)))
    check("物流方式缺失=72", int(o["物流方式"].isna().sum()) == 72)

    # 会员人均消费应随等级递增（修正后的业务直觉）
    per = valid.groupby("会员等级")["订单金额"].sum() / cu["会员等级"].value_counts()
    order = ["普通", "银牌", "金牌", "钻石"]
    vals = [per.get(k, 0) for k in order]
    check("会员人均消费随等级递增", vals[0] < vals[1] < vals[2] < vals[3],
          " < ".join(f"{k}:{v:,.0f}" for k, v in zip(order, vals)))


def main() -> int:
    verify_saas()
    verify_retail()
    verify_ads()
    verify_service()
    verify_ecommerce()

    head("汇总")
    failed = [c for c in _checks if not c[1]]
    print(f"共 {len(_checks)} 项断言，通过 {len(_checks) - len(failed)}，失败 {len(failed)}")
    if failed:
        print("\n失败项:")
        for name, _, detail in failed:
            print(f"  - {name}  {detail}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())