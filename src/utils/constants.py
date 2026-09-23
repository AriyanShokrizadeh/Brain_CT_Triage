"""Shared application constants."""

HEMORRHAGE_OUTPUTS = {
    "EpiduralHemorrhage": "V_EDH",
    "SubduralHemorrhage": "V_SDH",
    "IntraparenchymalHemorrhage": "V_IPH",
    "SubarachnoidHemorrhage": "V_SAH",
    "IntraventricularHemorrhage": "V_IVH",
}

MLS_KEYPOINTS = (
    "AnteriorFalxAttachment",
    "PosteriorFalxAttachment",
    "OutermostPointOfTheFalx",
)

SUBMISSION_COLUMNS = (
    "series_id",
    "V_EDH",
    "V_SDH",
    "V_IPH",
    "V_SAH",
    "V_IVH",
    "fracture_prob",
    "MLS_mm",
)

DINO_WEIGHTS = "backbone.pt"
MULTITASK_WEIGHTS = "best_model.pt"
