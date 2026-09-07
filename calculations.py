import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Dict, List, Any, Optional, Tuple


def calculate_cash_flows(inputs: Dict[str, Any]) -> Dict[str, Any]:
    initial_investment = float(inputs["initial_investment"])
    project_life = int(float(inputs["project_life"]))
    annual_revenue = float(inputs["annual_revenue"])
    operating_costs = float(inputs["operating_costs"])
    tax_rate = float(inputs["tax_rate"])
    working_capital = float(inputs.get("working_capital", 0))
    terminal_value = float(inputs.get("terminal_value", 0))
    wacc = float(inputs["wacc"])
    financing_rate = float(inputs.get("financing_rate", wacc))
    reinvestment_rate = float(inputs.get("reinvestment_rate", wacc))
    growth_rate = float(inputs.get("growth_rate", 0))
    depreciation_rate = float(inputs.get("depreciation_rate", 0))
    currency = inputs.get("currency", "USD")

    depreciation = initial_investment * depreciation_rate if depreciation_rate > 0 else initial_investment / project_life

    years = list(range(0, project_life + 1))
    cash_flows = []
    discount_factors = []
    present_values = []
    cumulative_cash_flows = []

    cumulative = 0.0

    for year in years:
        if year == 0:
            cf = -(initial_investment + working_capital)
            cash_flows.append(cf)
            discount_factors.append(1.0)
            present_values.append(cf)
            cumulative += cf
            cumulative_cash_flows.append(cumulative)
        else:
            growth = (1 + growth_rate) ** (year - 1)
            rev = annual_revenue * growth
            costs = operating_costs * growth
            ebit = rev - costs - depreciation
            taxes = ebit * tax_rate if ebit > 0 else 0
            net_income = ebit - taxes
            ocf = net_income + depreciation

            if year == project_life:
                ocf += working_capital + terminal_value

            cash_flows.append(ocf)
            df = 1 / (1 + wacc) ** year
            discount_factors.append(df)
            pv = ocf * df
            present_values.append(pv)
            cumulative += ocf
            cumulative_cash_flows.append(cumulative)

    total_pv_inflows = sum(present_values[1:])
    npv = sum(present_values)
    pi = total_pv_inflows / initial_investment if initial_investment != 0 else 0
    roi = (sum(cash_flows[1:]) - initial_investment) / initial_investment if initial_investment != 0 else 0

    irr = calculate_irr(cash_flows)
    mirr = calculate_mIRR(cash_flows, financing_rate, reinvestment_rate)
    payback = calculate_payback(cash_flows)

    holding_period_return = sum(cash_flows[1:]) / initial_investment if initial_investment != 0 else 0
    annualized_return = (1 + holding_period_return) ** (1 / project_life) - 1 if project_life > 0 else 0

    dcf_value = total_pv_inflows + (terminal_value / (1 + wacc) ** project_life) if terminal_value > 0 else total_pv_inflows

    cash_flow_table = pd.DataFrame({
        "Year": years,
        "Cash Flow": cash_flows,
        "Discount Factor": discount_factors,
        "Present Value": present_values,
        "Cumulative Cash Flow": cumulative_cash_flows,
    })

    dcf_table = pd.DataFrame({
        "Year": years,
        "Free Cash Flow": cash_flows,
        "Discount Factor": discount_factors,
        "Present Value of Cash Flow": present_values,
    })

    return {
        "cash_flows": cash_flows,
        "years": years,
        "discount_factors": discount_factors,
        "present_values": present_values,
        "cumulative_cash_flows": cumulative_cash_flows,
        "cash_flow_table": cash_flow_table,
        "dcf_table": dcf_table,
        "npv": npv,
        "irr": irr,
        "mirr": mirr,
        "payback": payback,
        "pi": pi,
        "roi": roi,
        "total_pv_inflows": total_pv_inflows,
        "dcf_value": dcf_value,
        "holding_period_return": holding_period_return,
        "annualized_return": annualized_return,
        "initial_investment": initial_investment,
        "project_life": project_life,
        "wacc": wacc,
        "financing_rate": financing_rate,
        "reinvestment_rate": reinvestment_rate,
        "terminal_value": terminal_value,
        "working_capital": working_capital,
        "tax_rate": tax_rate,
        "depreciation": depreciation,
        "currency": currency,
    }


def calculate_irr(cash_flows: List[float]) -> float:
    if len(cash_flows) < 2:
        return 0.0

    positive = any(cf > 0 for cf in cash_flows[1:])
    negative = any(cf < 0 for cf in cash_flows)

    if not positive or not negative:
        if positive and cash_flows[0] >= 0:
            return 1.0
        return 0.0

    try:
        def npv_func(r):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

        irr = brentq(npv_func, -0.5, 10.0, maxiter=1000, xtol=1e-12)
        return irr
    except (ValueError, RuntimeError):
        try:
            rates = np.linspace(-0.49, 5.0, 10000)
            npvs = [sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows)) for r in rates]
            idx = np.argmin(np.abs(npvs))
            return float(rates[idx])
        except Exception:
            return 0.0


def calculate_mIRR(cash_flows: List[float], finance_rate: float, reinvestment_rate: float) -> float:
    n = len(cash_flows) - 1
    if n <= 0:
        return 0.0

    pv_negatives = 0.0
    fv_positives = 0.0

    for t, cf in enumerate(cash_flows):
        if cf < 0:
            pv_negatives += cf / (1 + finance_rate) ** t
        elif cf > 0:
            fv_positives += cf * (1 + reinvestment_rate) ** (n - t)

    if pv_negatives == 0 or fv_positives <= 0:
        return 0.0

    pv_negatives = abs(pv_negatives)

    try:
        mirr = (fv_positives / pv_negatives) ** (1.0 / n) - 1
        if np.isnan(mirr) or np.isinf(mirr):
            return 0.0
        return mirr
    except (ZeroDivisionError, ValueError):
        return 0.0


def calculate_payback(cash_flows: List[float]) -> float:
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0 and i > 0:
            prev_cumulative = cumulative - cf
            if cf != 0:
                fraction = -prev_cumulative / cf
            else:
                fraction = 0
            return (i - 1) + fraction

    if cumulative < 0:
        return float("inf")
    return len(cash_flows)


def get_metric_status(metric_name: str, value: float, threshold: float,
                      higher_is_better: bool = True) -> Dict[str, str]:
    from data_validation import explain_metric, format_currency, format_pct

    if higher_is_better:
        favorable = value >= threshold
    else:
        favorable = value <= threshold

    decision = "ACCEPT" if favorable else "REJECT"
    result = explain_metric(metric_name, value, threshold, higher_is_better)

    return {
        "value": value,
        "threshold": threshold,
        "decision": decision,
        "reason": result["reason"],
        "favorable": favorable,
    }


def calculate_all_metrics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    cf_result = calculate_cash_flows(inputs)

    wacc = cf_result["wacc"]
    project_life = cf_result["project_life"]

    npv_status = get_metric_status("NPV", cf_result["npv"], 0, higher_is_better=True)
    irr_status = get_metric_status("IRR", cf_result["irr"], wacc, higher_is_better=True)
    mirr_status = get_metric_status("MIRR", cf_result["mirr"], wacc, higher_is_better=True)
    pi_status = get_metric_status("PI", cf_result["pi"], 1.0, higher_is_better=True)
    payback_status = get_metric_status("Payback", cf_result["payback"], project_life, higher_is_better=False)
    roi_status = get_metric_status("ROI", cf_result["roi"], 0, higher_is_better=True)

    cf_result["npv_status"] = npv_status
    cf_result["irr_status"] = irr_status
    cf_result["mirr_status"] = mirr_status
    cf_result["pi_status"] = pi_status
    cf_result["payback_status"] = payback_status
    cf_result["roi_status"] = roi_status

    cf_result["irr_status"]["wacc"] = wacc
    cf_result["mirr_status"]["wacc"] = wacc

    return cf_result


def calculate_sensitivity(inputs: Dict[str, Any], variable: str,
                          min_change: float = -0.30, max_change: float = 0.30,
                          steps: int = 13) -> Dict[str, Any]:
    base_value = float(inputs.get(variable, 0))
    if base_value == 0 and variable not in ("growth_rate",):
        base_value = 1.0

    variations = np.linspace(min_change, max_change, steps)
    results = []

    for var in variations:
        modified = dict(inputs)
        modified[variable] = base_value * (1 + var) if base_value != 0 else var

        if variable == "tax_rate" or variable == "wacc" or variable == "financing_rate" or variable == "reinvestment_rate":
            modified[variable] = base_value + var
            modified[variable] = max(0, min(modified[variable], 1.0))

        try:
            metrics = calculate_all_metrics(modified)
            results.append({
                "variation": var,
                "variation_pct": f"{var:+.0%}",
                "npv": metrics["npv"],
                "irr": metrics["irr"],
                "mirr": metrics["mirr"],
                "pi": metrics["pi"],
                "payback": metrics["payback"],
                "roi": metrics["roi"],
            })
        except Exception:
            continue

    df = pd.DataFrame(results)

    npv_range = 0
    if len(df) > 0:
        npv_range = df["npv"].max() - df["npv"].min()

    return {
        "variable": variable,
        "base_value": base_value,
        "results": df,
        "npv_range": npv_range,
        "variations": variations.tolist(),
    }


def calculate_full_sensitivity(inputs: Dict[str, Any]) -> Dict[str, Any]:
    variables = ["wacc", "annual_revenue", "operating_costs", "initial_investment"]
    growth_rate = inputs.get("growth_rate", 0)
    if growth_rate:
        variables.append("growth_rate")

    sensitivities = {}
    for var in variables:
        if var in inputs:
            sensitivities[var] = calculate_sensitivity(inputs, var)

    ranked = sorted(sensitivities.items(), key=lambda x: x[1]["npv_range"], reverse=True)
    ranking = [{"variable": v, "npv_range": s["npv_range"]} for v, s in ranked]

    return {
        "sensitivities": sensitivities,
        "ranking": ranking,
        "most_sensitive": ranking[0]["variable"] if ranking else None,
        "least_sensitive": ranking[-1]["variable"] if ranking else None,
    }
