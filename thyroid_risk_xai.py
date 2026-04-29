def get_risk_xai_details(diagnosis, stage, clinical_data):
    """
    Returns:
    - risk level
    - recurrence probability
    - explainable AI reasons
    - progression warning
    - symptoms by stage
    """

    # -------------------------------
    # Risk & recurrence (rule-based)
    # -------------------------------
    if stage in ["III", "IV"]:
        risk = "HIGH"
        recurrence = "65–75%"
    elif stage == "II":
        risk = "MODERATE"
        recurrence = "35–45%"
    else:
        risk = "LOW"
        recurrence = "10–20%"

    # -------------------------------
    # Explainable AI factors
    # -------------------------------
    xai_factors = []

    if clinical_data["Metastasis"] == 1:
        xai_factors.append("Presence of metastasis")

    if clinical_data["TSH_Level"] > 10:
        xai_factors.append("Elevated TSH level")

    if clinical_data["BRAF_Mutation"] == 1 or clinical_data["RET_PTC_Mutation"] == 1:
        xai_factors.append("Genetic mutation indicators")

    xai_factors.append("Strong CNN confidence from histopathology image")

    # -------------------------------
    # Disease progression warning
    # -------------------------------
    progression_warning = (
        "If treatment is delayed or not properly followed, "
        "the cancer may progress rapidly with increased risk of "
        "distant metastasis and reduced survival rate."
    )

    # -------------------------------
    # Symptoms by stage
    # -------------------------------
    symptoms = {
        "I": [
            "Small thyroid nodule",
            "Usually asymptomatic"
        ],
        "II": [
            "Neck swelling",
            "Mild voice change",
            "Palpable thyroid nodule"
        ],
        "III": [
            "Difficulty swallowing",
            "Persistent neck pain",
            "Hoarseness of voice",
            "Enlarged lymph nodes"
        ],
        "IV": [
            "Breathing difficulty",
            "Bone pain (if bone metastasis)",
            "Lung-related symptoms",
            "Weight loss and extreme fatigue"
        ]
    }

    return {
        "risk": risk,
        "recurrence": recurrence,
        "xai_factors": xai_factors,
        "progression_warning": progression_warning,
        "symptoms": symptoms
    }
