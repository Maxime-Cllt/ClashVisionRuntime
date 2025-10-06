//! Utility functions for detection operations.

use super::bbox::BoundingBox;

/// Filters bounding boxes by confidence threshold.
pub fn filter_by_confidence(boxes: &[BoundingBox], threshold: f32) -> Vec<BoundingBox> {
    boxes
        .iter()
        .filter(|bbox| bbox.confidence >= threshold)
        .copied()
        .collect()
}

/// Groups bounding boxes by class ID.
pub fn group_by_class(boxes: &[BoundingBox]) -> std::collections::HashMap<usize, Vec<BoundingBox>> {
    let mut grouped = std::collections::HashMap::new();

    for bbox in boxes {
        grouped
            .entry(bbox.class_id)
            .or_insert_with(Vec::new)
            .push(*bbox);
    }

    grouped
}

/// Calculates basic statistics for a collection of bounding boxes.
#[derive(Debug)]
pub struct DetectionStats {
    pub total_detections: usize,
    pub classes_detected: std::collections::HashSet<usize>,
    pub average_confidence: f32,
    pub confidence_range: (f32, f32),
}

impl DetectionStats {
    pub fn from_boxes(boxes: &[BoundingBox]) -> Self {
        if boxes.is_empty() {
            return Self {
                total_detections: 0,
                classes_detected: std::collections::HashSet::new(),
                average_confidence: 0.0,
                confidence_range: (0.0, 0.0),
            };
        }

        let total_detections = boxes.len();
        let classes_detected = boxes.iter().map(|b| b.class_id).collect();
        let confidences: Vec<f32> = boxes.iter().map(|b| b.confidence).collect();

        let average_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let min_confidence = confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_confidence = confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            total_detections,
            classes_detected,
            average_confidence,
            confidence_range: (min_confidence, max_confidence),
        }
    }
}
