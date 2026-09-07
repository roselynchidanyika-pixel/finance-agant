import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculations import calculate_all_metrics
from data_validation import validate_project_inputs
from risk_analysis import assess_risks
from scenario_analysis import run_scenario_analysis


def test_profitable_project():
    print("\n" + "=" * 70)
    print("TEST 1: Normal Profitable Project")
    print("=" * 70)

    inputs = {
        "project_name": "Profitable Manufacturing Plant",
        "project_description": "New factory for consumer goods",
        "initial_investment": 1000000,
        "project_life": 10,
        "annual_revenue": 500000,
        "operating_costs": 200000,
        "tax_rate": 0.25,
        "working_capital": 50000,
        "terminal_value": 200000,
        "wacc": 0.10,
        "financing_rate": 0.08,
        "reinvestment_rate": 0.06,
        "growth_rate": 0.03,
        "depreciation_rate": 0.10,
        "currency": "USD",
    }

    is_valid, errors, warnings = validate_project_inputs(inputs)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  Error: {e}")

    metrics = calculate_all_metrics(inputs)
    risk_data = assess_risks(inputs, metrics)

    print(f"\nNPV: ${metrics['npv']:,.0f}")
    print(f"IRR: {metrics['irr']:.1%}")
    print(f"MIRR: {metrics['mirr']:.1%}")
    print(f"PI: {metrics['pi']:.2f}")
    print(f"ROI: {metrics['roi']:.1%}")
    print(f"Payback: {metrics['payback']:.1f} years")
    print(f"Risk Level: {risk_data['overall_level']}")
    print(f"Decision: {metrics['npv_status']['decision']}")

    expected = "ACCEPT"
    actual = metrics["npv_status"]["decision"]
    passed = actual == expected
    print(f"\nExpected: {expected}")
    print(f"Actual: {actual}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    assert passed, f"Test 1 failed: expected {expected}, got {actual}"
    return passed


def test_negative_npv_project():
    print("\n" + "=" * 70)
    print("TEST 2: Negative NPV Project")
    print("=" * 70)

    inputs = {
        "project_name": "Unprofitable Venture",
        "project_description": "High cost, low return project",
        "initial_investment": 5000000,
        "project_life": 5,
        "annual_revenue": 400000,
        "operating_costs": 350000,
        "tax_rate": 0.30,
        "working_capital": 200000,
        "terminal_value": 0,
        "wacc": 0.15,
        "financing_rate": 0.12,
        "reinvestment_rate": 0.08,
        "growth_rate": 0.02,
        "depreciation_rate": 0.10,
        "currency": "USD",
    }

    is_valid, errors, warnings = validate_project_inputs(inputs)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")

    metrics = calculate_all_metrics(inputs)
    risk_data = assess_risks(inputs, metrics)

    print(f"\nNPV: ${metrics['npv']:,.0f}")
    print(f"IRR: {metrics['irr']:.1%}")
    print(f"MIRR: {metrics['mirr']:.1%}")
    print(f"PI: {metrics['pi']:.2f}")
    print(f"ROI: {metrics['roi']:.1%}")
    print(f"Payback: {metrics['payback']:.1f} years" if metrics['payback'] != float('inf') else "Payback: Never")
    print(f"Risk Level: {risk_data['overall_level']}")
    print(f"Decision: {metrics['npv_status']['decision']}")

    expected = "REJECT"
    actual = metrics["npv_status"]["decision"]
    passed = actual == expected
    print(f"\nExpected: {expected}")
    print(f"Actual: {actual}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    assert passed, f"Test 2 failed: expected {expected}, got {actual}"
    return passed


def test_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 3: Edge Cases")
    print("=" * 70)

    print("\n--- Edge Case 3a: Very Small Investment ---")
    inputs_small = {
        "project_name": "Micro Project",
        "project_description": "Very small investment test",
        "initial_investment": 100,
        "project_life": 3,
        "annual_revenue": 200,
        "operating_costs": 50,
        "tax_rate": 0.25,
        "working_capital": 0,
        "terminal_value": 0,
        "wacc": 0.10,
        "financing_rate": 0.08,
        "reinvestment_rate": 0.06,
        "growth_rate": 0.0,
        "depreciation_rate": 0.33,
        "currency": "USD",
    }
    try:
        metrics_small = calculate_all_metrics(inputs_small)
        print(f"  NPV: ${metrics_small['npv']:,.0f}")
        print(f"  Decision: {metrics_small['npv_status']['decision']}")
        print("  RESULT: PASS (no crash)")
    except Exception as e:
        print(f"  RESULT: FAIL - {e}")
        return False

    print("\n--- Edge Case 3b: Zero Revenue ---")
    inputs_zero = {
        "project_name": "Zero Revenue Test",
        "project_description": "Testing zero revenue handling",
        "initial_investment": 1000000,
        "project_life": 5,
        "annual_revenue": 0,
        "operating_costs": 50000,
        "tax_rate": 0.25,
        "working_capital": 0,
        "terminal_value": 0,
        "wacc": 0.10,
        "financing_rate": 0.08,
        "reinvestment_rate": 0.06,
        "growth_rate": 0.0,
        "depreciation_rate": 0.20,
        "currency": "USD",
    }
    try:
        metrics_zero = calculate_all_metrics(inputs_zero)
        print(f"  NPV: ${metrics_zero['npv']:,.0f}")
        print(f"  Decision: {metrics_zero['npv_status']['decision']}")
        print("  RESULT: PASS (no crash)")
    except Exception as e:
        print(f"  RESULT: FAIL - {e}")
        return False

    print("\n--- Edge Case 3c: Validation with Missing Fields ---")
    invalid_inputs = {"project_name": "Incomplete"}
    is_valid, errors, warnings = validate_project_inputs(invalid_inputs)
    passed_valid = not is_valid and len(errors) > 0
    print(f"  Valid: {is_valid}")
    print(f"  Errors caught: {len(errors)}")
    print(f"  RESULT: {'PASS' if passed_valid else 'FAIL'}")

    print("\n--- Edge Case 3d: Negative Investment ---")
    neg_inputs = {
        "project_name": "Neg Test",
        "project_description": "Test",
        "initial_investment": -100000,
        "project_life": 5,
        "annual_revenue": 50000,
        "operating_costs": 20000,
        "tax_rate": 0.25,
        "working_capital": 0,
        "terminal_value": 0,
        "wacc": 0.10,
        "financing_rate": 0.08,
        "reinvestment_rate": 0.06,
    }
    is_valid_neg, errors_neg, _ = validate_project_inputs(neg_inputs)
    passed_neg = not is_valid_neg
    print(f"  Valid: {is_valid_neg}")
    print(f"  Errors caught: {len(errors_neg)}")
    print(f"  RESULT: {'PASS' if passed_neg else 'FAIL'}")

    return True


def test_high_risk_project():
    print("\n" + "=" * 70)
    print("TEST 4: High-Risk / Worst-Case Project")
    print("=" * 70)

    inputs = {
        "project_name": "High Risk Speculative Venture",
        "project_description": "Speculative project with thin margins and high WACC",
        "initial_investment": 10000000,
        "project_life": 7,
        "annual_revenue": 2000000,
        "operating_costs": 1900000,
        "tax_rate": 0.35,
        "working_capital": 1000000,
        "terminal_value": 500000,
        "wacc": 0.20,
        "financing_rate": 0.18,
        "reinvestment_rate": 0.10,
        "growth_rate": 0.02,
        "depreciation_rate": 0.14,
        "currency": "ZAR",
    }

    is_valid, errors, warnings = validate_project_inputs(inputs)
    print(f"Validation: {'PASS' if is_valid else 'FAIL (expected)'}")
    if warnings:
        for w in warnings:
            print(f"  Warning: {w}")

    metrics = calculate_all_metrics(inputs)
    risk_data = assess_risks(inputs, metrics)

    print(f"\nNPV: ${metrics['npv']:,.0f}")
    print(f"IRR: {metrics['irr']:.1%}")
    print(f"MIRR: {metrics['mirr']:.1%}")
    print(f"PI: {metrics['pi']:.2f}")
    print(f"ROI: {metrics['roi']:.1%}")
    print(f"Payback: {metrics['payback']:.1f} years" if metrics['payback'] != float('inf') else "Payback: Never")
    print(f"Risk Level: {risk_data['overall_level']}")

    scenario_data = run_scenario_analysis(inputs)
    print(f"\nBest Case NPV: ${scenario_data['best_case']['npv']:,.0f}")
    print(f"Base Case NPV: ${scenario_data['base_case']['npv']:,.0f}")
    print(f"Worst Case NPV: ${scenario_data['worst_case']['npv']:,.0f}")

    overall_metrics_reject = (
        metrics["npv_status"]["decision"] == "REJECT" or
        risk_data["overall_level"] in ("HIGH", "VERY HIGH") or
        scenario_data["worst_case"]["npv_status"]["decision"] == "REJECT"
    )
    passed = overall_metrics_reject
    actual = "REJECT" if metrics["npv_status"]["decision"] == "REJECT" else "REVIEW"
    print(f"\nExpected: REJECT or REVIEW")
    print(f"Actual: {actual}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    assert passed, f"Test 4 failed"
    return passed


def run_all_tests():
    print("\n" + "#" * 70)
    print("#  FINANCIAL ENGINEERING INVESTMENT DECISION AGENT - TEST SUITE")
    print("#" * 70)

    results = {}
    tests = [
        ("Test 1: Normal Profitable Project", test_profitable_project),
        ("Test 2: Negative NPV Project", test_negative_npv_project),
        ("Test 3: Edge Cases", test_edge_cases),
        ("Test 4: High-Risk Project", test_high_risk_project),
    ]

    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "PASS" if passed else "FAIL"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"
        except Exception as e:
            results[name] = f"ERROR: {e}"

    print("\n" + "#" * 70)
    print("#  TEST RESULTS SUMMARY")
    print("#" * 70)
    all_passed = True
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  {name}: {result}")
        if result != "PASS":
            all_passed = False

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("#" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
