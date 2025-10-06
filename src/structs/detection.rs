/// Represents a detected object with its class ID, confidence score, and bounding box coordinates.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_creation() {
        let detection = Detection {
            class_id: 1,
            confidence: 0.85,
            bbox: [50.0, 50.0, 150.0, 150.0],
        };

        assert_eq!(detection.class_id, 1);
        assert!((detection.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(detection.bbox, [50.0, 50.0, 150.0, 150.0]);
    }

}
