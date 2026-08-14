def moderation_decision(label, confidence):

    # Low-confidence predictions should always be reviewed
    if confidence < 0.60:
        return {
            "action": "REVIEW",
            "message": "Low-confidence prediction. Send to a human moderator."
        }

    if label == "Hate Speech":

        if confidence >= 0.85:
            return {
                "action": "BLOCK",
                "message": "High-confidence hate speech detected."
            }

        return {
            "action": "REVIEW",
            "message": "Possible hate speech. Human review recommended."
        }

    elif label == "Offensive Language":

        if confidence >= 0.80:
            return {
                "action": "WARN",
                "message": "Offensive language detected. Warn the user."
            }

        return {
            "action": "REVIEW",
            "message": "Potential offensive content. Review recommended."
        }

    else:

        # Only allow safe content when confidence is high
        if confidence < 0.80:
            return {
                "action": "REVIEW",
                "message": "Content appears safe, but confidence is not high enough. Human review recommended."
            }

        return {
            "action": "ALLOW",
            "message": "No harmful content detected with high confidence."
        }