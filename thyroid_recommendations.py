# =====================================
# Thyroid Cancer Diagnostic Recommendations
# =====================================

def get_recommendation(diagnosis, stage):
    """
    diagnosis: 'PTC' or 'FTC'
    stage: 'I', 'II', 'III', 'IV'
    """

    recommendations = {

        "PTC": {
            "I": {
                "Diagnosis": "Papillary Thyroid Carcinoma – Stage I",
                "Recommendation": [
                    "Total or partial thyroidectomy",
                    "Active surveillance in low-risk patients",
                    "Regular ultrasound follow-up",
                    "TSH suppression therapy"
                ]
            },
            "II": {
                "Diagnosis": "Papillary Thyroid Carcinoma – Stage II",
                "Recommendation": [
                    "Total thyroidectomy",
                    "Radioactive iodine (RAI) therapy",
                    "TSH suppression therapy",
                    "Periodic thyroglobulin monitoring"
                ]
            },
            "III": {
                "Diagnosis": "Papillary Thyroid Carcinoma – Stage III",
                "Recommendation": [
                    "Total thyroidectomy with lymph node dissection",
                    "Radioactive iodine therapy",
                    "TSH suppression therapy",
                    "Close imaging surveillance"
                ]
            },
            "IV": {
                "Diagnosis": "Papillary Thyroid Carcinoma – Stage IV",
                "Recommendation": [
                    "Aggressive surgical resection",
                    "Radioactive iodine therapy",
                    "External beam radiation therapy",
                    "Targeted therapy or chemotherapy",
                    "Palliative care if needed"
                ]
            }
        },

        "FTC": {
            "I": {
                "Diagnosis": "Follicular Thyroid Carcinoma – Stage I",
                "Recommendation": [
                    "Total thyroidectomy",
                    "Routine follow-up and monitoring",
                    "TSH suppression therapy"
                ]
            },
            "II": {
                "Diagnosis": "Follicular Thyroid Carcinoma – Stage II",
                "Recommendation": [
                    "Total thyroidectomy",
                    "Radioactive iodine therapy",
                    "TSH suppression therapy"
                ]
            },
            "III": {
                "Diagnosis": "Follicular Thyroid Carcinoma – Stage III",
                "Recommendation": [
                    "Total thyroidectomy",
                    "Radioactive iodine therapy",
                    "Lymph node assessment",
                    "Long-term monitoring"
                ]
            },
            "IV": {
                "Diagnosis": "Follicular Thyroid Carcinoma – Stage IV",
                "Recommendation": [
                    "Aggressive surgical management",
                    "Radioactive iodine therapy",
                    "Systemic therapy (targeted drugs)",
                    "Palliative and supportive care"
                ]
            }
        }
    }

    return recommendations.get(diagnosis, {}).get(stage, {
        "Diagnosis": "Unknown",
        "Recommendation": ["No recommendation available"]
    })
