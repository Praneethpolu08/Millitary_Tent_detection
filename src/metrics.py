def generate_detection_summary(detections):

    total_detections = len(detections)

    avg_confidence = 0

    if total_detections > 0:

        avg_confidence = (
            sum(d["confidence"] for d in detections)
            / total_detections
        )

    return {
        "total_detections": total_detections,
        "average_confidence": avg_confidence
    }