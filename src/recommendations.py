"""Rule-based retention recommendations (not a machine learning model)."""

from __future__ import annotations


def generate_recommendations(customer: dict, probability: float, risk: str) -> list[str]:
    """Return 2-4 business actions based on profile and predicted risk."""
    recs: list[str] = []

    if risk == "HIGH" or probability >= 0.60:
        recs.append("Prioritize this customer for retention outreach.")
    elif risk == "MEDIUM":
        recs.append("Schedule a proactive check-in before the next billing cycle.")

    contract = str(customer.get("Contract", "")).strip()
    if contract == "Month-to-month":
        recs.append("Offer a discounted annual or two-year contract.")

    monthly = float(customer.get("MonthlyCharges", 0) or 0)
    if monthly >= 70:
        recs.append("Consider a personalized pricing or loyalty offer.")

    if str(customer.get("TechSupport", "")).strip() != "Yes":
        recs.append("Offer technical support assistance.")

    tenure = float(customer.get("tenure", 0) or 0)
    if tenure <= 12:
        recs.append("Provide onboarding support and early customer engagement.")

    if str(customer.get("OnlineSecurity", "")).strip() != "Yes":
        recs.append("Consider offering security-related services.")

    service_flags = [
        str(customer.get("PhoneService", "")).strip() == "Yes",
        str(customer.get("MultipleLines", "")).strip() == "Yes",
        str(customer.get("OnlineBackup", "")).strip() == "Yes",
        str(customer.get("DeviceProtection", "")).strip() == "Yes",
        str(customer.get("StreamingTV", "")).strip() == "Yes",
        str(customer.get("StreamingMovies", "")).strip() == "Yes",
    ]
    if sum(service_flags) >= 3:
        recs.append("Offer bundled service discounts.")

    if str(customer.get("PaymentMethod", "")).strip() == "Electronic check":
        recs.append("Encourage automatic payment (bank or card) to reduce billing friction.")

    internet = str(customer.get("InternetService", "")).strip()
    if internet == "Fiber optic":
        recs.append("Review fiber plan value and add complementary support services.")

    unique: list[str] = []
    for item in recs:
        if item not in unique:
            unique.append(item)
    if not unique:
        unique.append("Maintain regular service quality and monitor usage changes.")
    return unique[:4]
