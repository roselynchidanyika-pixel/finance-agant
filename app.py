import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_validation import validate_project_inputs, validate_uploaded_data, get_default_inputs, format_currency, format_pct
from calculations import calculate_all_metrics, calculate_sensitivity, calculate_full_sensitivity
from risk_analysis import assess_risks, get_risk_summary
from scenario_analysis import run_scenario_analysis, explain_scenario
from report_generator import generate_management_report
from email_service import send_report_email, build_email_body

st.set_page_config(
    page_title="Integrated Investment Decision Agent for Capital Projects",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric > div { background-color: #f8f9fa; padding: 10px; border-radius: 8px; }
    .decision-accept { color: #0a7d0a; font-weight: bold; font-size: 1.3em; }
    .decision-reject { color: #c41e3a; font-weight: bold; font-size: 1.3em; }
    .decision-review { color: #c49b00; font-weight: bold; font-size: 1.3em; }
    .metric-box { background: #f0f4f8; padding: 15px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #0066cc; }
    .risk-high { color: #c41e3a; font-weight: bold; }
    .risk-medium { color: #c49b00; font-weight: bold; }
    .risk-low { color: #0a7d0a; font-weight: bold; }
    h1 { color: #003366; }
    h2 { color: #004d80; }
    h3 { color: #006699; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_fx_rates():
    rates = {"USD": 1.0}
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("result") == "success":
                api_rates = data.get("rates", {})
                if "ZAR" in api_rates:
                    rates["ZAR"] = api_rates["ZAR"]
    except Exception:
        pass

    if "ZAR" not in rates:
        rates["ZAR"] = 18.50

    try:
        resp2 = requests.get("https://open.er-api.com/v6/latest/ZAR", timeout=10)
        if resp2.status_code == 200:
            data2 = resp2.json()
            if data2.get("result") == "success":
                api_rates2 = data2.get("rates", {})
                if "USD" in api_rates2:
                    rates["ZAR"] = 1.0 / api_rates2["USD"] if api_rates2["USD"] != 0 else rates.get("ZAR", 18.50)
    except Exception:
        pass

    if "ZIG" not in rates:
        rates["ZIG"] = 13.50

    rates["USD_ZAR"] = rates["ZAR"]
    rates["USD_ZIG"] = rates["ZIG"]
    rates["ZIG_ZAR"] = rates["ZAR"] / rates["ZIG"] if rates["ZIG"] != 0 else 0
    rates["ZAR_USD"] = 1.0 / rates["ZAR"] if rates["ZAR"] != 0 else 0
    rates["ZIG_USD"] = 1.0 / rates["ZIG"] if rates["ZIG"] != 0 else 0
    rates["ZAR_ZIG"] = rates["ZIG"] / rates["ZAR"] if rates["ZAR"] != 0 else 0

    return rates


def convert_currency(amount, from_currency, to_currency, rates):
    if from_currency == to_currency:
        return amount
    usd_amount = amount / rates.get(from_currency, 1) if from_currency != "USD" else amount
    return usd_amount * rates.get(to_currency, 1)


def get_final_decision(metrics, risk_data, scenario_data):
    accept_score = 0
    reject_score = 0
    reasons_for = []
    reasons_against = []

    if metrics["npv_status"]["decision"] == "ACCEPT":
        accept_score += 3
        reasons_for.append("NPV is positive, indicating the project creates value above the required return.")
    else:
        reject_score += 3
        reasons_against.append("NPV is negative, indicating value destruction.")

    if metrics["irr_status"]["decision"] == "ACCEPT":
        accept_score += 2
        reasons_for.append(f"IRR ({metrics['irr']:.1%}) exceeds WACC ({metrics['wacc']:.1%}).")
    else:
        reject_score += 2
        reasons_against.append(f"IRR ({metrics['irr']:.1%}) is below WACC ({metrics['wacc']:.1%}).")

    if metrics["mirr_status"]["decision"] == "ACCEPT":
        accept_score += 2
        reasons_for.append(f"MIRR ({metrics['mirr']:.1%}) exceeds the required return.")
    else:
        reject_score += 2
        reasons_against.append(f"MIRR ({metrics['mirr']:.1%}) is below the required return.")

    if metrics["pi_status"]["decision"] == "ACCEPT":
        accept_score += 1
        reasons_for.append(f"Profitability Index ({metrics['pi']:.2f}) is above 1.0.")
    else:
        reject_score += 1
        reasons_against.append(f"Profitability Index ({metrics['pi']:.2f}) is below 1.0.")

    if metrics["payback_status"]["decision"] == "ACCEPT":
        accept_score += 1
        reasons_for.append("Payback period is within the project life.")
    else:
        reject_score += 1
        reasons_against.append("Payback period exceeds the project life.")

    if metrics["roi_status"]["decision"] == "ACCEPT":
        accept_score += 1
        reasons_for.append("ROI is positive.")
    else:
        reject_score += 1
        reasons_against.append("ROI is negative.")

    risk_level = risk_data.get("overall_level", "MEDIUM")
    if risk_level in ("LOW",):
        accept_score += 1
        reasons_for.append(f"Risk level is {risk_level}.")
    elif risk_level in ("HIGH", "VERY HIGH"):
        reject_score += 1
        reasons_against.append(f"Risk level is {risk_level}.")

    worst_case = scenario_data.get("worst_case", {})
    if worst_case.get("npv_status", {}).get("decision") == "REJECT":
        reject_score += 1
        reasons_against.append("Worst-case scenario produces negative NPV.")
    else:
        accept_score += 1
        reasons_for.append("Even the worst-case scenario remains viable.")

    total = accept_score + reject_score
    accept_ratio = accept_score / total if total > 0 else 0.5

    if accept_ratio >= 0.65:
        decision = "ACCEPT"
    elif accept_ratio <= 0.35:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    if decision == "ACCEPT":
        recommendation = (
            "The project demonstrates strong financial fundamentals across multiple metrics. "
            "Management should proceed with the investment, subject to ongoing monitoring of "
            "key assumptions and regular performance reviews against projections."
        )
    elif decision == "REJECT":
        recommendation = (
            "Do not proceed under the current assumptions. Management should reconsider project costs, "
            "expected revenues, financing structure, or required return before reassessment. "
            "Alternative projects with better risk-adjusted returns should be evaluated."
        )
    else:
        recommendation = (
            "The project presents mixed signals. Management should conduct additional due diligence, "
            "gather more market data, and consider a phased investment approach. "
            "Key assumptions should be stress-tested further before a final commitment."
        )

    all_reasons = []
    for r in reasons_for:
        all_reasons.append(f"[+] {r}")
    for r in reasons_against:
        all_reasons.append(f"[-] {r}")

    return {
        "decision": decision,
        "accept_score": accept_score,
        "reject_score": reject_score,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "supporting_points": all_reasons,
        "reason": "; ".join(reasons_for[:2] + reasons_against[:2]) if (reasons_for or reasons_against) else "Insufficient data.",
        "recommendation": recommendation,
    }


def create_cash_flow_chart(metrics):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Annual Cash Flows", "Cumulative Cash Flow"),
        vertical_spacing=0.12,
    )
    years = metrics["years"]
    cash_flows = metrics["cash_flows"]
    colors = ["#c41e3a" if cf < 0 else "#0a7d0a" for cf in cash_flows]

    fig.add_trace(
        go.Bar(x=years, y=cash_flows, name="Cash Flow", marker_color=colors,
               text=[format_currency(cf) for cf in cash_flows], textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=years, y=metrics["cumulative_cash_flows"], name="Cumulative CF",
                   mode="lines+markers", line=dict(color="#0066cc", width=2)),
        row=2, col=1,
    )
    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Cash Flow", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative CF", row=2, col=1)
    return fig


def create_pv_chart(metrics):
    fig = go.Figure()
    years = metrics["years"]
    pvs = metrics["present_values"]
    colors = ["#c41e3a" if pv < 0 else "#0066cc" for pv in pvs]
    fig.add_trace(go.Bar(
        x=years, y=pvs, name="Present Value", marker_color=colors,
        text=[format_currency(pv) for pv in pvs], textposition="outside",
    ))
    fig.update_layout(title="Present Value of Cash Flows", template="plotly_white",
                      xaxis_title="Year", yaxis_title="Present Value", height=400)
    return fig


def create_scenario_chart(scenario_data):
    fig = go.Figure()
    labels = ["Best Case", "Base Case", "Worst Case"]
    npvs = [scenario_data["best_case"]["npv"], scenario_data["base_case"]["npv"],
            scenario_data["worst_case"]["npv"]]
    colors = ["#0a7d0a", "#0066cc", "#c41e3a"]

    fig.add_trace(go.Bar(
        x=labels, y=npvs, name="NPV", marker_color=colors,
        text=[format_currency(n) for n in npvs], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Scenario Analysis - NPV Comparison", template="plotly_white",
                      yaxis_title="NPV", height=400)
    return fig


def create_sensitivity_tornado(sensitivity_data):
    ranking = sensitivity_data.get("ranking", [])
    if not ranking:
        return None

    variables = [item["variable"] for item in ranking]
    ranges = [item["npv_range"] for item in ranking]

    fig = go.Figure(go.Bar(
        x=ranges, y=variables, orientation="h",
        marker_color=["#c41e3a" if i == 0 else "#0066cc" for i in range(len(variables))],
        text=[f"${r:,.0f}" for r in ranges], textposition="outside",
    ))
    fig.update_layout(title="Sensitivity Analysis - Tornado Chart", template="plotly_white",
                      xaxis_title="NPV Range ($)", height=400)
    return fig


def create_sensitivity_line_chart(sensitivity_data, variable):
    sens = sensitivity_data.get("sensitivities", {}).get(variable)
    if sens is None or sens["results"].empty:
        return None

    df = sens["results"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["variation_pct"], y=df["npv"], mode="lines+markers",
        name="NPV", line=dict(color="#0066cc", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="NPV = 0")
    fig.update_layout(
        title=f"NPV Sensitivity to {variable}",
        template="plotly_white",
        xaxis_title=f"Change in {variable}",
        yaxis_title="NPV ($)",
        height=400,
    )
    return fig


def create_risk_radar(risk_data):
    risks = risk_data.get("risks", [])
    if not risks:
        return None

    categories = [r["name"] for r in risks]
    values = [r["severity_score"] for r in risks]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        line=dict(color="#c41e3a"),
        fillcolor="rgba(196, 30, 58, 0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title="Risk Profile Radar", template="plotly_white", height=500,
        showlegend=False,
    )
    return fig


def display_metric_with_explanation(name, value, status, currency="USD"):
    col1, col2 = st.columns([1, 2])
    with col1:
        if isinstance(value, float):
            if abs(value) > 1000:
                st.metric(label=name, value=format_currency(value, currency))
            elif abs(value) < 1:
                st.metric(label=name, value=format_pct(value))
            else:
                st.metric(label=name, value=f"{value:.2f}")
        else:
            st.metric(label=name, value=str(value))

        decision = status.get("decision", "N/A")
        if decision == "ACCEPT":
            st.markdown(f'<span class="decision-accept">Decision: {decision}</span>', unsafe_allow_html=True)
        elif decision == "REJECT":
            st.markdown(f'<span class="decision-reject">Decision: {decision}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="decision-review">Decision: {decision}</span>', unsafe_allow_html=True)

    with col2:
        st.info(status.get("reason", "No explanation available."))


def sidebar_inputs():
    st.sidebar.header("Investment Proposal Inputs")
    st.sidebar.markdown("---")

    input_method = st.sidebar.radio(
        "Input Method",
        ["Manual Input", "Upload CSV/Excel"],
        horizontal=True,
    )

    data = get_default_inputs()

    if input_method == "Upload CSV/Excel":
        uploaded_file = st.sidebar.file_uploader("Upload Project Data", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                is_valid, errors, warnings, parsed = validate_uploaded_data(df)
                if warnings:
                    for w in warnings:
                        st.sidebar.warning(w)
                if not is_valid:
                    for e in errors:
                        st.sidebar.error(e)
                    return None
                data.update(parsed)
                st.sidebar.success("File uploaded and validated successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                return None
    else:
        st.sidebar.subheader("Project Details")
        data["project_name"] = st.sidebar.text_input("Project Name", value="New Investment Project")
        data["project_description"] = st.sidebar.text_area("Project Description", value="Investment project evaluation", height=68)
        data["currency"] = st.sidebar.selectbox("Currency", ["USD", "ZIG", "ZAR", "GBP", "EUR"], index=0)

        st.sidebar.subheader("Financial Parameters")
        data["initial_investment"] = st.sidebar.number_input("Initial Investment", min_value=0.0, value=1000000.0, step=10000.0, format="%.0f")
        data["project_life"] = st.sidebar.number_input("Project Life (years)", min_value=1, max_value=100, value=10, step=1)
        data["working_capital"] = st.sidebar.number_input("Working Capital", min_value=0.0, value=100000.0, step=5000.0, format="%.0f")
        data["terminal_value"] = st.sidebar.number_input("Terminal Value", min_value=0.0, value=0.0, step=10000.0, format="%.0f")

        st.sidebar.subheader("Revenue & Costs (Annual)")
        data["annual_revenue"] = st.sidebar.number_input("Annual Revenue", min_value=0.0, value=500000.0, step=10000.0, format="%.0f")
        data["operating_costs"] = st.sidebar.number_input("Operating Costs", min_value=0.0, value=200000.0, step=10000.0, format="%.0f")

        st.sidebar.subheader("Rates & Assumptions")
        data["tax_rate"] = st.sidebar.number_input("Tax Rate (%)", min_value=0.0, max_value=80.0, value=25.0, step=1.0) / 100
        data["wacc"] = st.sidebar.number_input("WACC / Discount Rate (%)", min_value=0.1, max_value=50.0, value=10.0, step=0.5) / 100
        data["financing_rate"] = st.sidebar.number_input("Financing Rate (%)", min_value=0.1, max_value=50.0, value=8.0, step=0.5) / 100
        data["reinvestment_rate"] = st.sidebar.number_input("Reinvestment Rate (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.5) / 100
        data["revenue_growth"] = st.sidebar.number_input("Revenue Growth (%/yr)", min_value=0.0, max_value=50.0, value=3.0, step=0.5) / 100
        data["cost_growth"] = st.sidebar.number_input("Cost Growth (%/yr)", min_value=0.0, max_value=50.0, value=3.0, step=0.5) / 100
        data["terminal_growth"] = st.sidebar.number_input("Terminal Growth (%/yr)", min_value=0.0, max_value=30.0, value=0.0, step=0.5) / 100
        data["depreciation_rate"] = st.sidebar.number_input("Depreciation Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0) / 100

    return data


def tab_executive_summary(metrics, risk_data, scenario_data, final_decision, inputs, fx_rates):
    st.header("Executive Summary")
    currency = inputs.get("currency", "USD")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        decision = final_decision["decision"]
        if decision == "ACCEPT":
            st.markdown(f'<div class="metric-box"><h3 style="color:#0a7d0a">DECISION: ACCEPT</h3></div>', unsafe_allow_html=True)
        elif decision == "REJECT":
            st.markdown(f'<div class="metric-box"><h3 style="color:#c41e3a">DECISION: REJECT</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-box"><h3 style="color:#c49b00">DECISION: REVIEW</h3></div>', unsafe_allow_html=True)
    with col2:
        st.metric("Investment", format_currency(metrics["initial_investment"], currency))
    with col3:
        st.metric("NPV", format_currency(metrics["npv"], currency))
    with col4:
        st.metric("Risk Level", risk_data["overall_level"])

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("IRR", format_pct(metrics["irr"]))
    with col6:
        st.metric("MIRR", format_pct(metrics["mirr"]))
    with col7:
        st.metric("ROI", format_pct(metrics["roi"]))
    with col8:
        pb = metrics["payback"]
        st.metric("Payback", f"{pb:.1f} yrs" if pb != float("inf") else "N/A")

    st.markdown("---")
    st.subheader("Decision Rationale")
    for point in final_decision.get("supporting_points", []):
        if point.startswith("[+]"):
            st.markdown(f" :green[{point}]")
        else:
            st.markdown(f" :red[{point}]")

    st.markdown("---")
    st.subheader("Investment in Multiple Currencies")
    inv = float(inputs.get("initial_investment", 0))
    from_currency = currency

    fx_cols = st.columns(3)
    for i, target in enumerate(["USD", "ZIG", "ZAR"]):
        with fx_cols[i]:
            converted = convert_currency(inv, from_currency, target, fx_rates)
            rate_key = f"{from_currency}_{target}"
            rate_val = fx_rates.get(rate_key, "N/A")
            st.metric(
                f"Amount in {target}",
                f"{converted:,.0f} {target}",
                f"Rate: {rate_val:.4f}" if isinstance(rate_val, (int, float)) else str(rate_val),
            )

    st.info(
        "Exchange rates are indicative and sourced from public APIs. "
        "ZIG (Zimbabwe Gold) rates are estimates and may differ from actual transaction rates. "
        "Use these comparisons to assess currency advantages for your investment."
    )


def tab_capital_budgeting(metrics, inputs):
    st.header("Capital Budgeting Analysis")
    currency = inputs.get("currency", "USD")

    st.subheader("Cash Flow Analysis")
    cf_table = metrics["cash_flow_table"].copy()

    format_cols = [c for c in cf_table.columns if c not in ("Year", "Initial Investment", "Working Capital", "Terminal Value")]
    for c in format_cols:
        if c in cf_table.columns and c != "Year":
            cf_table[c] = cf_table[c].apply(lambda x: format_currency(x, currency) if isinstance(x, (int, float)) else x)

    st.dataframe(cf_table, use_container_width=True, hide_index=True)

    st.plotly_chart(create_cash_flow_chart(metrics), use_container_width=True)

    st.subheader("Capital Budgeting Metrics")
    display_metric_with_explanation("Net Present Value (NPV)", metrics["npv"], metrics["npv_status"], currency)
    display_metric_with_explanation("Internal Rate of Return (IRR)", metrics["irr"], metrics["irr_status"], currency)
    display_metric_with_explanation("Modified IRR (MIRR)", metrics["mirr"], metrics["mirr_status"], currency)
    display_metric_with_explanation("Profitability Index (PI)", metrics["pi"], metrics["pi_status"], currency)
    pb_text = f"{metrics['payback']:.1f} years" if metrics['payback'] != float('inf') else "Never"
    display_metric_with_explanation("Payback Period", metrics["payback"], metrics["payback_status"], currency)
    display_metric_with_explanation("Return on Investment (ROI)", metrics["roi"], metrics["roi_status"], currency)


def tab_dcf(metrics, inputs):
    st.header("DCF Valuation")
    currency = inputs.get("currency", "USD")

    st.subheader("Discounted Cash Flow Table")
    dcf_table = metrics["dcf_table"].copy()
    dcf_table["Free Cash Flow"] = dcf_table["Free Cash Flow"].apply(lambda x: format_currency(x, currency))
    dcf_table["Discount Factor"] = dcf_table["Discount Factor"].apply(lambda x: f"{x:.4f}")
    dcf_table["Present Value"] = dcf_table["Present Value"].apply(lambda x: format_currency(x, currency))
    dcf_table["Cumulative Present Value"] = dcf_table["Cumulative Present Value"].apply(lambda x: format_currency(x, currency))
    st.dataframe(dcf_table, use_container_width=True, hide_index=True)

    st.plotly_chart(create_pv_chart(metrics), use_container_width=True)

    st.subheader("DCF Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total PV of Cash Flows", format_currency(metrics["total_pv_inflows"], currency))
        if metrics["terminal_value"] > 0:
            pv_tv = metrics["terminal_value"] / (1 + metrics["wacc"]) ** metrics["project_life"]
            st.metric("PV of Terminal Value", format_currency(pv_tv, currency))
        st.metric("DCF Value", format_currency(metrics["dcf_value"], currency))
    with col2:
        st.metric("Initial Investment", format_currency(metrics["initial_investment"], currency))
        net_value = metrics["dcf_value"] - metrics["initial_investment"]
        st.metric("Net Value Created", format_currency(net_value, currency))

    st.subheader("Interpretation")
    if metrics["dcf_value"] >= metrics["initial_investment"]:
        st.success(
            f"The DCF value of **{format_currency(metrics['dcf_value'], currency)}** exceeds the initial investment "
            f"of **{format_currency(metrics['initial_investment'], currency)}**. The project creates "
            f"**{format_currency(net_value, currency)}** in value after accounting for the time value of money."
        )
    else:
        st.error(
            f"The DCF value of **{format_currency(metrics['dcf_value'], currency)}** is below the initial investment "
            f"of **{format_currency(metrics['initial_investment'], currency)}**. The project does not generate "
            f"sufficient present value to justify the investment, resulting in a value shortfall of "
            f"**{format_currency(abs(net_value), currency)}**."
        )


def tab_returns(metrics, inputs):
    st.header("Returns Analysis")
    currency = inputs.get("currency", "USD")

    st.subheader("IRR Analysis")
    display_metric_with_explanation("Internal Rate of Return", metrics["irr"], metrics["irr_status"], currency)
    st.caption(f"WACC: {format_pct(metrics['wacc'])}")

    st.subheader("MIRR Analysis")
    display_metric_with_explanation("Modified IRR", metrics["mirr"], metrics["mirr_status"], currency)
    st.caption(f"Financing Rate: {format_pct(metrics['financing_rate'])} | Reinvestment Rate: {format_pct(metrics['reinvestment_rate'])}")

    st.subheader("ROI Analysis")
    display_metric_with_explanation("Return on Investment", metrics["roi"], metrics["roi_status"], currency)

    st.subheader("Additional Returns Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Holding Period Return", format_pct(metrics["holding_period_return"]))
        st.info(
            "The Holding Period Return represents the total return earned over the entire project life "
            "without accounting for the time value of money."
        )
    with col2:
        st.metric("Annualized Return", format_pct(metrics["annualized_return"]))
        st.info(
            "The Annualized Return converts the total project return into an equivalent annual rate, "
            "allowing comparison with annual returns of other investments."
        )

    st.subheader("Profitability Index")
    display_metric_with_explanation("Profitability Index", metrics["pi"], metrics["pi_status"], currency)
    st.caption(
        f"PV of Inflows: {format_currency(metrics['total_pv_inflows'], currency)} | "
        f"Investment: {format_currency(metrics['initial_investment'], currency)}"
    )

    st.subheader("Payback Period")
    display_metric_with_explanation("Payback Period", metrics["payback"], metrics["payback_status"], currency)
    st.caption(f"Project Life: {int(metrics['project_life'])} years")


def tab_risk(risk_data):
    st.header("Risk Analysis")
    st.metric("Overall Risk Level", f"{risk_data['overall_level']} ({risk_data['overall_score']:.1f}/10)")

    radar_fig = create_risk_radar(risk_data)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)

    for risk in risk_data["risks"]:
        severity = risk["severity"]
        impact_text = risk.get("impact_text", risk.get("impact", ""))
        with st.expander(f"{risk['name']} - {severity}", expanded=(severity in ("HIGH", "VERY HIGH", "MODERATE"))):
            st.markdown(
                f"**{risk['name']}:** {severity} | **Impact:** {impact_text}"
            )
            st.write(risk["explanation"])
            st.markdown("**Possible mitigation:** " + risk["mitigation"])


def tab_scenario(scenario_data, inputs):
    st.header("Scenario Analysis")
    currency = inputs.get("currency", "USD")

    st.plotly_chart(create_scenario_chart(scenario_data), use_container_width=True)

    for label, key in [("Best Case", "best_case"), ("Base Case", "base_case"), ("Worst Case", "worst_case")]:
        scenario = scenario_data[key]
        with st.expander(f"{label} - Decision: {scenario['npv_status']['decision']}", expanded=(key == "base_case")):
            st.write(scenario.get("scenario_description", ""))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NPV", format_currency(scenario["npv"], currency))
                st.metric("IRR", format_pct(scenario["irr"]))
            with col2:
                st.metric("MIRR", format_pct(scenario["mirr"]))
                st.metric("PI", f"{scenario['pi']:.2f}")
            with col3:
                st.metric("ROI", format_pct(scenario["roi"]))
                pb = scenario["payback"]
                st.metric("Payback", f"{pb:.1f} yrs" if pb != float("inf") else "N/A")

            scenario_reason = scenario.get("scenario_reason", scenario["npv_status"]["reason"])
            decision = scenario["npv_status"]["decision"]
            if decision == "ACCEPT":
                st.success(f"**Decision: {decision}** - {scenario_reason}")
            elif decision == "REJECT":
                st.error(f"**Decision: {decision}** - {scenario_reason}")
            else:
                st.warning(f"**Decision: {decision}** - {scenario_reason}")


def tab_sensitivity(sensitivity_data, inputs):
    st.header("Sensitivity Analysis")

    ranking = sensitivity_data.get("ranking", [])

    if ranking:
        st.subheader("Sensitivity Summary")
        summary_rows = []
        for idx, item in enumerate(ranking):
            summary_rows.append({
                "Variable": item.get("label", item.get("variable", "")),
                "Id": item.get("variable", ""),
                "NPV at -30%": format_currency(item.get("npv_at_minus30", 0)),
                "NPV at Base": format_currency(item.get("npv_at_base", 0)),
                "NPV at +30%": format_currency(item.get("npv_at_plus30", 0)),
                "Spread": format_currency(item.get("npv_range", 0)),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if sensitivity_data.get("most_sensitive") and sensitivity_data.get("least_sensitive"):
            st.info(
                f"The most sensitive variable is **{sensitivity_data['most_sensitive']}** and the least "
                f"sensitive is **{sensitivity_data['least_sensitive']}**. Management should prioritise "
                f"review, hedging and control of the most sensitive variable because small deviations "
                f"from the assumption have the largest impact on project value and on the investment decision."
            )

    tornado = create_sensitivity_tornado(sensitivity_data)
    if tornado:
        st.plotly_chart(tornado, use_container_width=True)

    sens_map = sensitivity_data.get("sensitivities", {})

    st.subheader("Per-Variable Analysis")
    for item in ranking:
        var_name = item.get("variable", "")
        sens = sens_map.get(var_name)
        if sens is None:
            continue
        label = sens.get("label", var_name)
        with st.expander(f"{label}"):
            npv_minus = item.get("npv_at_minus30", 0)
            npv_base = item.get("npv_at_base", 0)
            npv_plus = item.get("npv_at_plus30", 0)

            st.markdown(
                f"NPV swings from {format_currency(npv_minus)} to {format_currency(npv_plus)} "
                f"around the base {format_currency(npv_base)}."
            )

            if var_name == "wacc":
                st.write(
                    "Raising the WACC reduces the present value of future cash flows, lowering NPV "
                    "(discounting effect); lowering it does the reverse. This reflects how expensive "
                    "the project's capital is."
                )
            elif var_name in ("annual_revenue",):
                st.write(
                    "Revenues are the primary inflow driver. Higher revenue raises cash flows and NPV; "
                    "lower revenue erodes them, and the effect compounds over the project life."
                )
            elif var_name in ("operating_costs",):
                st.write(
                    "Operating costs subtract directly from cash flows. Higher costs depress NPV and lower "
                    "costs improve it, with the swing persisting across every year of the project."
                )
            elif var_name in ("initial_investment",):
                st.write(
                    "The initial investment sets the baseline outlay. A larger outlay reduces NPV "
                    "one-for-one in present-value terms; a smaller one improves it."
                )
            elif var_name in ("revenue_growth",):
                st.write(
                    "Revenue growth compounds inflows over the project life. Higher growth raises terminal "
                    "and early-period cash flows, boosting NPV; lower growth reduces project value."
                )
            elif var_name in ("cost_growth",):
                st.write(
                    "Cost growth compounds outflows over the life. Higher cost growth erodes margins and NPV; "
                    "lower cost growth protects value."
                )
            else:
                st.write(
                    f"Changes in {label} affect project cash flows and thus NPV. The magnitude of impact "
                    f"depends on how central this assumption is to the project's economics."
                )

            line_chart = create_sensitivity_line_chart(sensitivity_data, var_name)
            if line_chart:
                st.plotly_chart(line_chart, use_container_width=True)

            df = sens["results"]
            if not df.empty:
                df_disp = df.copy()
                for c in ["npv", "payback"]:
                    if c in df_disp.columns:
                        df_disp[c] = df_disp[c].apply(
                            lambda x: format_currency(x) if c == "npv" else (f"{x:.1f}" if x != float("inf") else "Never")
                        )
                for c in ["irr", "mirr", "roi"]:
                    if c in df_disp.columns:
                        df_disp[c] = df_disp[c].apply(lambda x: format_pct(x))
                if "pi" in df_disp.columns:
                    df_disp["pi"] = df_disp["pi"].apply(lambda x: f"{x:.2f}")
                st.dataframe(df_disp, use_container_width=True, hide_index=True)


def tab_fx(inputs, fx_rates):
    st.header("FX Exchange Rate Analysis")
    st.subheader("Live Exchange Rates (ZIG / USD / ZAR)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("USD/ZAR", f"{fx_rates.get('USD_ZAR', 0):.4f}")
    with col2:
        st.metric("USD/ZIG", f"{fx_rates.get('USD_ZIG', 0):.4f}")
    with col3:
        st.metric("ZIG/ZAR", f"{fx_rates.get('ZIG_ZAR', 0):.4f}")
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M"))

    st.info(
        "ZIG (Zimbabwe Gold) rates are indicative estimates. Always verify current rates with your bank or "
        "authorized foreign exchange dealer before making investment decisions."
    )

    st.markdown("---")
    st.subheader("Currency Comparison for Your Investment")
    st.write("See how your investment amount translates across USD, ZIG, and ZAR:")

    currency = inputs.get("currency", "USD")
    inv = float(inputs.get("initial_investment", 0))

    if inv > 0:
        comp_data = []
        for target in ["USD", "ZIG", "ZAR"]:
            converted = convert_currency(inv, currency, target, fx_rates)
            comp_data.append({
                "Currency": target,
                "Amount": f"{converted:,.0f} {target}",
                "Description": f"Equivalent of {format_currency(inv, currency)} in {target}",
            })

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=["USD", "ZIG", "ZAR"],
            y=[convert_currency(inv, currency, t, fx_rates) for t in ["USD", "ZIG", "ZAR"]],
            marker_color=["#0066cc", "#c49b00", "#0a7d0a"],
            text=[f"{convert_currency(inv, currency, t, fx_rates):,.0f}" for t in ["USD", "ZIG", "ZAR"]],
            textposition="outside",
        ))
        fig.update_layout(title="Investment Amount Across Currencies", template="plotly_white",
                          yaxis_title="Amount", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Multi-Currency Scenario Comparison")
    st.write("Compare project returns across different investment currencies:")

    currency_options = ["USD", "ZIG", "ZAR"]
    selected_currencies = st.multiselect("Select currencies to compare", currency_options, default=currency_options)

    if selected_currencies and inv > 0:
        results = []
        for curr in selected_currencies:
            converted_inv = convert_currency(inv, currency, curr, fx_rates)
            converted_rev = convert_currency(float(inputs.get("annual_revenue", 0)), currency, curr, fx_rates)
            converted_costs = convert_currency(float(inputs.get("operating_costs", 0)), currency, curr, fx_rates)
            converted_wc = convert_currency(float(inputs.get("working_capital", 0)), currency, curr, fx_rates)
            converted_tv = convert_currency(float(inputs.get("terminal_value", 0)), currency, curr, fx_rates)

            mod_inputs = dict(inputs)
            mod_inputs["initial_investment"] = converted_inv
            mod_inputs["annual_revenue"] = converted_rev
            mod_inputs["operating_costs"] = converted_costs
            mod_inputs["working_capital"] = converted_wc
            mod_inputs["terminal_value"] = converted_tv
            mod_inputs["currency"] = curr

            try:
                mod_metrics = calculate_all_metrics(mod_inputs)
                results.append({
                    "Currency": curr,
                    "Investment": format_currency(converted_inv, curr),
                    "NPV": format_currency(mod_metrics["npv"], curr),
                    "IRR": format_pct(mod_metrics["irr"]),
                    "MIRR": format_pct(mod_metrics["mirr"]),
                    "PI": f"{mod_metrics['pi']:.2f}",
                    "ROI": format_pct(mod_metrics["roi"]),
                    "Decision": mod_metrics["npv_status"]["decision"],
                })
            except Exception as e:
                results.append({"Currency": curr, "Error": str(e)})

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Manual Currency Converter")
    man_amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0, format="%.0f")
    man_from = st.selectbox("From", ["USD", "ZIG", "ZAR"], index=0)
    man_to = st.selectbox("To", ["USD", "ZIG", "ZAR"], index=1)

    if man_amount > 0:
        converted = convert_currency(man_amount, man_from, man_to, fx_rates)
        st.success(f"**{man_amount:,.0f} {man_from} = {converted:,.2f} {man_to}**")
        rate_key = f"{man_from}_{man_to}"
        st.caption(f"Exchange rate: 1 {man_from} = {fx_rates.get(rate_key, 0):.4f} {man_to}")


def tab_final_decision(metrics, risk_data, scenario_data, final_decision, inputs):
    st.header("Final Investment Decision")
    currency = inputs.get("currency", "USD")
    decision = final_decision["decision"]

    if decision == "ACCEPT":
        st.markdown(f'<div style="background:#d4edda;border:2px solid #0a7d0a;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#0a7d0a;margin:0">ACCEPT</h1></div>', unsafe_allow_html=True)
    elif decision == "REJECT":
        st.markdown(f'<div style="background:#f8d7da;border:2px solid #c41e3a;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#c41e3a;margin:0">REJECT</h1></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#fff3cd;border:2px solid #c49b00;padding:20px;border-radius:10px;text-align:center">'
                    f'<h1 style="color:#c49b00;margin:0">REVIEW</h1></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Supporting Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("NPV", format_currency(metrics["npv"], currency))
        st.caption(f"Status: {metrics['npv_status']['decision']}")
    with col2:
        st.metric("IRR", format_pct(metrics["irr"]))
        st.caption(f"vs WACC: {format_pct(metrics['wacc'])}")
    with col3:
        st.metric("MIRR", format_pct(metrics["mirr"]))
        st.caption(f"Status: {metrics['mirr_status']['decision']}")
    with col4:
        st.metric("PI", f"{metrics['pi']:.2f}")
        st.caption(f"Status: {metrics['pi_status']['decision']}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("ROI", format_pct(metrics["roi"]))
        st.caption(f"Status: {metrics['roi_status']['decision']}")
    with col6:
        pb = metrics["payback"]
        st.metric("Payback", f"{pb:.1f} yrs" if pb != float("inf") else "N/A")
        st.caption(f"Status: {metrics['payback_status']['decision']}")
    with col7:
        st.metric("Risk Level", risk_data["overall_level"])
    with col8:
        worst_npv = scenario_data["worst_case"]["npv"]
        st.metric("Worst Case NPV", format_currency(worst_npv, currency))

    st.markdown("---")
    st.subheader("Decision Analysis")

    st.markdown("**Arguments in FAVOR of the project:**")
    for r in final_decision.get("reasons_for", []):
        st.markdown(f" :green[+] {r}")

    st.markdown("**Arguments AGAINST the project:**")
    for r in final_decision.get("reasons_against", []):
        st.markdown(f" :red[-] {r}")

    st.markdown("---")
    st.subheader("Analyst Recommendation")
    st.write(final_decision.get("recommendation", "No recommendation available."))


def tab_report(inputs, metrics, risk_data, scenario_data, sensitivity_data, fx_data, final_decision):
    st.header("Management Report")
    st.write("Generate a comprehensive professional management report in Word format.")

    if st.button("Generate Management Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                report_path = generate_management_report(
                    inputs, metrics, risk_data, scenario_data, sensitivity_data, fx_data, final_decision,
                )
                st.success("Report generated successfully!")

                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download Management Report (.docx)",
                        data=f.read(),
                        file_name=f"Management_Report_{inputs.get('project_name', 'Project').replace(' ', '_')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            except Exception as e:
                st.error(f"Error generating report: {e}")


def tab_email(inputs, metrics, final_decision):
    st.header("Email Management Report")
    st.write("Send the investment analysis via email with the management report attached.")

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = os.environ.get("SMTP_PORT", "")
    smtp_user = os.environ.get("SMTP_USER", os.environ.get("SMTP_USERNAME", ""))
    smtp_pass = os.environ.get("SMTP_PASS", os.environ.get("SMTP_PASSWORD", ""))
    smtp_from = os.environ.get("SMTP_FROM", "")

    with st.container(border=True):
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("SMTP_HOST", "OK" if smtp_host else "MISSING")
        col_b.metric("SMTP_PORT", "OK" if smtp_port else "MISSING")
        col_c.metric("SMTP_USER", "OK" if smtp_user else "MISSING")
        col_d.metric("SMTP_PASS", "OK" if smtp_pass else "MISSING")
        col_e.metric("SMTP_FROM", "OK" if smtp_from else "MISSING")

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, smtp_from]):
        st.warning(
            "SMTP environment variables are not fully configured. Set SMTP_HOST, SMTP_PORT, "
            "SMTP_USERNAME, SMTP_PASSWORD and SMTP_FROM before sending."
        )

    recipient = st.text_input("Recipient Email")
    subject = st.text_input("Subject", value=f"Investment Decision Report - {inputs.get('project_name', 'Project')}")

    default_body = build_email_body(
        project_name=inputs.get("project_name", "Project"),
        npv=metrics["npv"],
        irr=metrics["irr"],
        mirr=metrics["mirr"],
        roi=metrics["roi"],
        payback=metrics["payback"],
        risk_level="LOW",
        decision=final_decision["decision"],
        main_reason=final_decision.get("reason", ""),
        currency=inputs.get("currency", "USD"),
    )
    message = st.text_area("Email Message", value=default_body, height=300)

    if st.button("Send Email", type="primary"):
        if not recipient:
            st.error("Please enter a recipient email address.")
            return

        report_path = None
        try:
            report_path = generate_management_report(
                inputs, metrics, {}, {}, {}, None, final_decision,
            )
        except Exception:
            pass

        result = send_report_email(
            recipient=recipient,
            subject=subject,
            body=message,
            attachment_path=report_path,
        )

        if result["success"]:
            st.success(result["message"])
        else:
            st.error(result["message"])


def tab_tests():
    st.header("Test Suite")
    st.write("Run automated test cases to verify calculation accuracy.")

    if st.button("Run All Tests", type="primary"):
        from test_cases import run_all_tests
        import io as _io
        import contextlib

        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            success = run_all_tests()

        output = buf.getvalue()
        st.code(output, language=None)

        if success:
            st.success("All tests passed!")
        else:
            st.error("Some tests failed. Check the output above for details.")


def tab_assumptions():
    st.header("Assumptions, Data Sources & Financial Logic")

    st.subheader("Assumptions")
    assumptions = [
        "Depreciation is calculated using the straight-line method over the project life.",
        "Tax is applied to EBIT (Earnings Before Interest and Taxes).",
        "Operating cash flows are calculated as: (Revenue - Costs) * (1 - Tax Rate) + Depreciation * Tax Rate.",
        "Working capital is invested at Year 0 and recovered at the end of the project.",
        "Terminal value is a lump sum provided by the user, received at the end of the project life.",
        "Revenue grows at the specified growth rate from Year 2 onwards.",
        "Costs grow at the same rate as revenue.",
        "Exchange rates are fetched from public APIs and are indicative only.",
        "ZIG (Zimbabwe Gold) exchange rates are estimates and may differ from actual rates.",
        "Scenario analysis uses fixed adjustments: Best (+20% revenue, -10% costs, -2% WACC), Worst (-20% revenue, +10% costs, +2% WACC).",
        "Risk assessment is based on analytical indicators derived from input data, not external market data.",
    ]
    for a in assumptions:
        st.markdown(f"- {a}")

    st.subheader("Data Sources")
    st.markdown("""
    - **User-Provided Data:** All financial projections, rates, and project parameters
    - **Uploaded Data:** CSV/Excel files provided by the user
    - **Exchange Rates:** Live rates from open.er-api.com (public API)
    - **Model Assumptions:** Depreciation method, tax treatment, cash flow timing
    """, unsafe_allow_html=True)

    st.subheader("Financial Logic")
    st.markdown("""
    | Metric | Formula | Decision Rule |
    |--------|---------|---------------|
    | **NPV** | Sum of discounted cash flows minus initial investment | ACCEPT if NPV >= 0 |
    | **IRR** | Rate where NPV = 0 | ACCEPT if IRR >= WACC |
    | **MIRR** | (FV of positive CFs / PV of negative CFs)^(1/n) - 1 | ACCEPT if MIRR >= WACC |
    | **PI** | PV of future cash flows / Initial investment | ACCEPT if PI >= 1.0 |
    | **Payback** | Time to recover initial investment | ACCEPT if Payback <= Project Life |
    | **ROI** | (Total returns - Investment) / Investment | ACCEPT if ROI >= 0 |
    | **Final Decision** | Weighted scoring across all metrics | Weighted score determines ACCEPT/REJECT/REVIEW |
    """, unsafe_allow_html=True)


def main():
    st.title("Integrated Investment Decision Agent for Capital Projects")
    st.markdown(
        "*Industry-independent capital budgeting, DCF, risk, scenario and decision engine.*"
    )
    st.markdown(
        "**Workflow:** Investment Proposal → Capital Budgeting → DCF → Returns → Risk → "
        "Scenario & Sensitivity → Final Decision → Management Report → Email"
    )
    st.markdown(
        "This is a professional analysis tool. Every metric is calculated, compared against a "
        "threshold, converted into a decision and explained in plain financial language."
    )
    st.markdown("---")

    fx_rates = fetch_fx_rates()

    inputs = sidebar_inputs()

    if inputs is None:
        st.info("Please enter project data or upload a file to begin analysis.")
        st.markdown("### Quick Start")
        st.markdown("1. Use the sidebar to enter project details manually, or upload a CSV/Excel file")
        st.markdown("2. Click **Analyze** to run the full investment analysis")
        st.markdown("3. Explore results across all tabs")
        st.markdown("4. Download the management report or email it to stakeholders")

        if st.button("Load Sample Project", type="primary"):
            sample = {
                "project_name": "Sample Manufacturing Plant",
                "project_description": "New manufacturing facility for consumer goods production",
                "initial_investment": 5000000,
                "project_life": 10,
                "annual_revenue": 2000000,
                "operating_costs": 800000,
                "tax_rate": 0.25,
                "working_capital": 500000,
                "terminal_value": 1000000,
                "wacc": 0.10,
                "financing_rate": 0.08,
                "reinvestment_rate": 0.06,
                "growth_rate": 0.03,
                "depreciation_rate": 0.10,
                "currency": "USD",
            }
            st.session_state["loaded_inputs"] = sample
            st.rerun()
        return

    if "loaded_inputs" in st.session_state:
        inputs = st.session_state.pop("loaded_inputs")

    is_valid, errors, warnings = validate_project_inputs(inputs)

    if warnings:
        for w in warnings:
            st.sidebar.warning(w)

    if not is_valid:
        for e in errors:
            st.sidebar.error(e)
        st.error("Please fix the input errors before running the analysis.")
        return

    st.sidebar.markdown("---")
    analyze = st.sidebar.button("Analyze Investment", type="primary", use_container_width=True)

    if not analyze and "metrics" not in st.session_state:
        st.info("Enter project data in the sidebar and click **Analyze Investment** to run the analysis.")
        return

    if analyze or "metrics" not in st.session_state:
        with st.spinner("Running comprehensive financial analysis..."):
            try:
                metrics = calculate_all_metrics(inputs)
                risk_data = assess_risks(inputs, metrics)
                scenario_data = run_scenario_analysis(inputs)
                sensitivity_data = calculate_full_sensitivity(inputs)
                final_decision = get_final_decision(metrics, risk_data, scenario_data)

                st.session_state["metrics"] = metrics
                st.session_state["risk_data"] = risk_data
                st.session_state["scenario_data"] = scenario_data
                st.session_state["sensitivity_data"] = sensitivity_data
                st.session_state["final_decision"] = final_decision
                st.session_state["inputs"] = inputs
                st.session_state["fx_rates"] = fx_rates
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return

    metrics = st.session_state["metrics"]
    risk_data = st.session_state["risk_data"]
    scenario_data = st.session_state["scenario_data"]
    sensitivity_data = st.session_state["sensitivity_data"]
    final_decision = st.session_state["final_decision"]

    fx_data = {"rates": fx_rates, "comparisons": []}
    inv = float(inputs.get("initial_investment", 0))
    for curr in ["USD", "ZIG", "ZAR"]:
        converted = convert_currency(inv, inputs.get("currency", "USD"), curr, fx_rates)
        fx_data["comparisons"].append({
            "currency": curr,
            "investment_amount": f"{converted:,.0f} {curr}",
            "equivalent_usd": f"${convert_currency(inv, inputs.get('currency', 'USD'), 'USD', fx_rates):,.0f}",
        })

    tab_names = [
        "Executive Summary", "Capital Budgeting", "DCF Valuation",
        "Returns Analysis", "Risk Analysis", "Scenario Analysis",
        "Sensitivity Analysis", "FX Currency Analysis", "Final Decision",
        "Management Report", "Email Report", "Test Cases", "Assumptions & Documentation",
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_executive_summary(metrics, risk_data, scenario_data, final_decision, inputs, fx_rates)
    with tabs[1]:
        tab_capital_budgeting(metrics, inputs)
    with tabs[2]:
        tab_dcf(metrics, inputs)
    with tabs[3]:
        tab_returns(metrics, inputs)
    with tabs[4]:
        tab_risk(risk_data)
    with tabs[5]:
        tab_scenario(scenario_data, inputs)
    with tabs[6]:
        tab_sensitivity(sensitivity_data, inputs)
    with tabs[7]:
        tab_fx(inputs, fx_rates)
    with tabs[8]:
        tab_final_decision(metrics, risk_data, scenario_data, final_decision, inputs)
    with tabs[9]:
        tab_report(inputs, metrics, risk_data, scenario_data, sensitivity_data, fx_data, final_decision)
    with tabs[10]:
        tab_email(inputs, metrics, final_decision)
    with tabs[11]:
        tab_tests()
    with tabs[12]:
        tab_assumptions()


if __name__ == "__main__":
    main()
