from typing import Dict, List, Any
import numpy as np


def assess_risks(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    risks = []

    revenue_risk = assess_revenue_risk(inputs, metrics)
    costs_risk = assess_cost_risk(inputs, metrics)
    wacc_risk = assess_wacc_risk(inputs, metrics)
    inflation_risk = assess_inflation_risk(inputs)
    exchange_risk = assess_exchange_risk(inputs)
    liquidity_risk = assess_liquidity_risk(inputs, metrics)
    recovery_risk = assess_recovery_risk(inputs, metrics)
    cashflow_risk = assess_cashflow_risk(inputs, metrics)

    risks = [revenue_risk, costs_risk, wacc_risk, inflation_risk,
             exchange_risk, liquidity_risk, recovery_risk, cashflow_risk]

    overall_score = np.mean([r["severity_score"] for r in risks])

    if overall_score <= 3:
        overall_level = "LOW"
    elif overall_score <= 5:
        overall_level = "MEDIUM"
    elif overall_score <= 7:
        overall_level = "HIGH"
    else:
        overall_level = "VERY HIGH"

    return {
        "risks": risks,
        "overall_score": overall_score,
        "overall_level": overall_level,
    }


def _make_risk(name: str, impact: str, severity: str, score: int,
               explanation: str, mitigation: str) -> Dict[str, Any]:
    return {
        "name": name,
        "impact": impact,
        "severity": severity,
        "severity_score": score,
        "explanation": explanation,
        "mitigation": mitigation,
    }


def assess_revenue_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    revenue = float(inputs.get("annual_revenue", 0))
    costs = float(inputs.get("operating_costs", 0))
    investment = float(inputs.get("initial_investment", 0))

    if investment == 0:
        return _make_risk("Revenue Risk", "N/A", "MEDIUM", 5,
                          "Unable to assess revenue risk without investment data.",
                          "Provide complete financial data for accurate risk assessment.")

    margin = (revenue - costs) / revenue if revenue > 0 else -1

    if margin < 0.1:
        severity, score = "HIGH", 8
        explanation = (
            f"Revenue risk is HIGH. The operating margin is only {margin:.1%}, meaning a small decline "
            f"in revenue or increase in costs could lead to operating losses. Revenue streams appear "
            f"vulnerable to market fluctuations."
        )
        mitigation = (
            "Diversify revenue streams, secure long-term contracts, build revenue reserves, "
            "and implement flexible pricing strategies."
        )
    elif margin < 0.3:
        severity, score = "MEDIUM", 5
        explanation = (
            f"Revenue risk is MEDIUM. The operating margin of {margin:.1%} provides moderate buffer, "
            f"but significant revenue declines could still impact project viability."
        )
        mitigation = (
            "Monitor market conditions regularly, maintain cost flexibility, "
            "and develop contingency plans for revenue shortfalls."
        )
    else:
        severity, score = "LOW", 3
        explanation = (
            f"Revenue risk is LOW. The operating margin of {margin:.1%} provides a healthy buffer "
            f"against revenue fluctuations."
        )
        mitigation = "Continue monitoring market conditions and maintain competitive positioning."

    return _make_risk("Revenue Risk", "Cash Flow & Profitability", severity, score, explanation, mitigation)


def assess_cost_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    revenue = float(inputs.get("annual_revenue", 0))
    costs = float(inputs.get("operating_costs", 0))

    if revenue == 0:
        return _make_risk("Operating Cost Risk", "N/A", "MEDIUM", 5,
                          "Unable to assess cost risk without revenue data.",
                          "Provide complete financial data.")

    cost_ratio = costs / revenue

    if cost_ratio > 0.8:
        severity, score = "HIGH", 7
        explanation = (
            f"Operating cost risk is HIGH. Costs represent {cost_ratio:.1%} of revenue, leaving minimal "
            f"margin for cost overruns. Any increase in operating costs could result in losses."
        )
        mitigation = (
            "Implement strict cost controls, negotiate long-term supplier contracts, "
            "invest in efficiency improvements, and build cost contingency reserves."
        )
    elif cost_ratio > 0.6:
        severity, score = "MEDIUM", 5
        explanation = (
            f"Operating cost risk is MEDIUM. Costs are {cost_ratio:.1%} of revenue. "
            f"Moderate cost increases could erode profitability."
        )
        mitigation = "Monitor costs regularly and maintain cost flexibility where possible."
    else:
        severity, score = "LOW", 2
        explanation = (
            f"Operating cost risk is LOW. Costs are {cost_ratio:.1%} of revenue, "
            f"providing substantial margin against cost increases."
        )
        mitigation = "Maintain current cost management practices."

    return _make_risk("Operating Cost Risk", "Profitability & Cash Flow", severity, score, explanation, mitigation)


def assess_wacc_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    wacc = float(inputs.get("wacc", 0.1))
    irr = metrics.get("irr", 0)
    spread = irr - wacc

    if spread < -0.02:
        severity, score = "HIGH", 8
        explanation = (
            f"WACC/Interest Rate risk is HIGH. The IRR ({irr:.1%}) is significantly below WACC ({wacc:.1%}). "
            f"Any increase in the cost of capital would further reduce project viability. "
            f"The project is highly sensitive to changes in interest rates."
        )
        mitigation = (
            "Consider fixing financing rates, refinancing at lower rates, "
            "reducing leverage, or restructuring the capital mix."
        )
    elif spread < 0.01:
        severity, score = "MEDIUM", 5
        explanation = (
            f"WACC/Interest Rate risk is MEDIUM. The spread between IRR ({irr:.1%}) and WACC ({wacc:.1%}) "
            f"is thin. Small increases in the cost of capital could turn the project unprofitable."
        )
        mitigation = "Lock in current financing rates and monitor market conditions closely."
    else:
        severity, score = "LOW", 2
        explanation = (
            f"WACC/Interest Rate risk is LOW. The healthy spread between IRR ({irr:.1%}) and "
            f"WACC ({wacc:.1%}) provides a buffer against interest rate increases."
        )
        mitigation = "Maintain current financing structure and periodic review."

    return _make_risk("WACC / Interest Rate Risk", "Discount Rate & Returns", severity, score, explanation, mitigation)


def assess_inflation_risk(inputs: Dict[str, Any]) -> Dict[str, Any]:
    revenue = float(inputs.get("annual_revenue", 0))
    costs = float(inputs.get("operating_costs", 0))
    investment = float(inputs.get("initial_investment", 0))

    cost_intensity = costs / investment if investment > 0 else 0.5

    if cost_intensity > 0.5:
        severity, score = "MEDIUM", 5
        explanation = (
            "Inflation risk is MEDIUM. High operating costs relative to the investment suggest that "
            "inflationary pressure on input costs could erode margins over the project's life."
        )
        mitigation = "Include inflation escalators in contracts, consider inflation-linked pricing."
    else:
        severity, score = "LOW", 3
        explanation = (
            "Inflation risk is LOW. The project's cost structure provides moderate protection "
            "against inflationary pressures."
        )
        mitigation = "Monitor inflation trends and adjust forecasts periodically."

    return _make_risk("Inflation Risk", "Cost Structure & Margins", severity, score, explanation, mitigation)


def assess_exchange_risk(inputs: Dict[str, Any]) -> Dict[str, Any]:
    currency = inputs.get("currency", "USD")

    if currency.upper() in ("ZIG", "ZAR"):
        severity, score = "HIGH", 7
        explanation = (
            f"Exchange rate risk is HIGH. The project is denominated in {currency}, which is subject to "
            f"exchange rate volatility. Significant currency fluctuations could reduce the project's value "
            f"when converted to other currencies, potentially impacting returns for international investors."
        )
        mitigation = (
            "Consider hedging strategies, maintain multi-currency reserves, "
            "structure revenues and costs in the same currency where possible, "
            "and monitor exchange rates closely."
        )
    else:
        severity, score = "LOW", 3
        explanation = (
            f"Exchange rate risk is LOW. The project is denominated in {currency}, a relatively "
            f"stable international currency. However, if any costs or revenues are in other currencies, "
            f"exposure should be evaluated."
        )
        mitigation = "Review any foreign currency exposures and hedge where necessary."

    return _make_risk("Exchange Rate Risk", "Currency Conversion & International Value", severity, score, explanation, mitigation)


def assess_liquidity_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    working_capital = float(inputs.get("working_capital", 0))
    initial_investment = float(inputs.get("initial_investment", 1))
    wc_ratio = working_capital / initial_investment if initial_investment > 0 else 0

    annual_cf = 0
    cash_flows = metrics.get("cash_flows", [])
    if len(cash_flows) > 1:
        annual_cf = cash_flows[1]

    if annual_cf < 0:
        severity, score = "HIGH", 8
        explanation = (
            "Liquidity risk is HIGH. The project generates negative operating cash flows in early years, "
            "which could strain liquidity and require additional funding."
        )
        mitigation = "Secure adequate working capital facilities and establish contingency funding arrangements."
    elif wc_ratio < 0.05:
        severity, score = "MEDIUM", 5
        explanation = (
            "Liquidity risk is MEDIUM. Working capital relative to the investment is low. "
            "Cash flow timing mismatches could cause short-term liquidity issues."
        )
        mitigation = "Ensure adequate working capital reserves and establish credit facilities."
    else:
        severity, score = "LOW", 2
        explanation = (
            "Liquidity risk is LOW. The project has adequate working capital provisions "
            "and generates positive cash flows."
        )
        mitigation = "Maintain current working capital management practices."

    return _make_risk("Liquidity Risk", "Cash Availability & Timing", severity, score, explanation, mitigation)


def assess_recovery_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    payback = metrics.get("payback", float("inf"))
    project_life = float(inputs.get("project_life", 10))

    if payback == float("inf") or payback > project_life:
        severity, score = "HIGH", 8
        explanation = (
            f"Investment recovery risk is HIGH. The payback period ({payback:.1f} years) exceeds "
            f"the project life ({int(project_life)} years). The initial investment is not recovered "
            f"within the expected operating period, exposing the investor to prolonged capital at risk."
        )
        mitigation = (
            "Reduce initial investment, accelerate revenue generation, "
            "or extend the project's operating life if feasible."
        )
    elif payback > project_life * 0.7:
        severity, score = "MEDIUM", 5
        explanation = (
            f"Investment recovery risk is MEDIUM. The payback period ({payback:.1f} years) is "
            f"close to the project life ({int(project_life)} years), leaving limited time for "
            f"profit generation after recovery."
        )
        mitigation = "Focus on accelerating revenue streams and managing costs to shorten payback."
    else:
        severity, score = "LOW", 2
        explanation = (
            f"Investment recovery risk is LOW. The payback period ({payback:.1f} years) is well "
            f"within the project life ({int(project_life)} years), allowing sufficient time for "
            f"profit generation after investment recovery."
        )
        mitigation = "Maintain current trajectory and reinvest recovered capital strategically."

    return _make_risk("Investment Recovery Risk", "Capital Recovery Timeline", severity, score, explanation, mitigation)


def assess_cashflow_risk(inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    cash_flows = metrics.get("cash_flows", [])
    if len(cash_flows) < 3:
        return _make_risk("Cash Flow Risk", "N/A", "MEDIUM", 5,
                          "Insufficient data to assess cash flow risk.",
                          "Provide more project years for analysis.")

    operating_cfs = cash_flows[1:]
    cf_std = np.std(operating_cfs) if len(operating_cfs) > 1 else 0
    cf_mean = np.mean(operating_cfs) if operating_cfs else 1
    cv = cf_std / abs(cf_mean) if cf_mean != 0 else 0

    negative_cfs = sum(1 for cf in operating_cfs if cf < 0)

    if cv > 0.5 or negative_cfs > 0:
        severity, score = "HIGH", 7
        explanation = (
            f"Cash flow risk is HIGH. Operating cash flows show high variability "
            f"(coefficient of variation: {cv:.2f})"
        )
        if negative_cfs > 0:
            explanation += f" with {negative_cfs} year(s) of negative cash flow"
        explanation += ". This creates uncertainty about the project's ability to meet financial obligations."
        mitigation = "Build cash reserves, secure backup funding, and implement rigorous cash flow monitoring."
    elif cv > 0.2:
        severity, score = "MEDIUM", 5
        explanation = (
            f"Cash flow risk is MEDIUM. Cash flows show moderate variability "
            f"(coefficient of variation: {cv:.2f}), suggesting some uncertainty in projected cash generation."
        )
        mitigation = "Maintain cash reserves and monitor actual vs. projected cash flows closely."
    else:
        severity, score = "LOW", 2
        explanation = (
            f"Cash flow risk is LOW. Cash flows are relatively stable "
            f"(coefficient of variation: {cv:.2f}), indicating predictable cash generation."
        )
        mitigation = "Continue current cash management practices."

    return _make_risk("Cash Flow Risk", "Cash Flow Stability & Predictability", severity, score, explanation, mitigation)


def get_risk_summary(risk_data: Dict[str, Any]) -> str:
    overall = risk_data["overall_level"]
    score = risk_data["overall_score"]

    high_risks = [r for r in risk_data["risks"] if r["severity"] == "HIGH"]
    med_risks = [r for r in risk_data["risks"] if r["severity"] == "MEDIUM"]
    low_risks = [r for r in risk_data["risks"] if r["severity"] == "LOW"]

    lines = [
        f"**Overall Risk Level: {overall}** (Score: {score:.1f}/10)",
        f"- High risks: {len(high_risks)}",
        f"- Medium risks: {len(med_risks)}",
        f"- Low risks: {len(low_risks)}",
        "",
    ]

    if high_risks:
        lines.append("**Key Concerns:**")
        for r in high_risks:
            lines.append(f"- {r['name']}: {r['explanation'][:100]}...")

    return "\n".join(lines)
