//! Non-Maximum Suppression implementation

use super::bbox::BoundingBox;

/// Performs Non-Maximum Suppression (NMS) on a list of bounding boxes.
///
/// # Arguments
/// * `boxes` - Slice of bounding boxes to filter
/// * `iou_threshold` - `IoU` threshold for suppression (typically 0.4-0.5)
///
/// # Returns
/// Vector of filtered bounding boxes
#[must_use]
pub fn nms(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort by confidence in descending order
    let mut sorted_boxes = boxes.to_vec();
    sorted_boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = Vec::with_capacity(boxes.len());
    let mut suppressed = vec![false; sorted_boxes.len()];

    for (i, current_box) in sorted_boxes.iter().enumerate() {
        if suppressed[i] {
            continue;
        }

        result.push(*current_box);

        // Suppress overlapping boxes
        for (j, other_box) in sorted_boxes.iter().enumerate().skip(i + 1) {
            if !suppressed[j] && current_box.iou(other_box) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    result
}

/// Performs class-agnostic NMS
#[must_use]
pub fn nms_class_agnostic(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    nms(boxes, iou_threshold)
}

/// Performs per-class NMS
#[must_use]
pub fn nms_per_class(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    use std::collections::HashMap;

    let mut class_boxes: HashMap<usize, Vec<BoundingBox>> = HashMap::new();

    // Group boxes by class
    for &bbox in boxes {
        class_boxes.entry(bbox.class_id).or_default().push(bbox);
    }

    let mut result = Vec::new();

    // Apply NMS per class
    for boxes_for_class in class_boxes.values() {
        result.extend(nms(boxes_for_class, iou_threshold));
    }

    // Sort final result by confidence
    result.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_empty() {
        let boxes = [];
        let result = nms(&boxes, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_single_box() {
        let boxes = [BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0, 0.9)];
        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_nms_suppression() {
        let boxes = [
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            BoundingBox::new(1.0, 1.0, 11.0, 11.0, 0, 0.8), // High overlap, should be suppressed
            BoundingBox::new(20.0, 20.0, 30.0, 30.0, 0, 0.7), // No overlap, should remain
        ];
        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].confidence, 0.9);
        assert_eq!(result[1].confidence, 0.7);
    }
}
