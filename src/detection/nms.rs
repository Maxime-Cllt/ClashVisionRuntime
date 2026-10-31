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

    // Sort indices by confidence (descending, stable) instead of cloning the boxes.
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        boxes[b]
            .confidence
            .partial_cmp(&boxes[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = Vec::with_capacity(boxes.len());
    let mut suppressed = vec![false; order.len()];

    for (i, &current_idx) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }

        let current_box = &boxes[current_idx];
        result.push(*current_box);

        // Suppress overlapping boxes
        for (j, &other_idx) in order.iter().enumerate().skip(i + 1) {
            if !suppressed[j] && current_box.iou(&boxes[other_idx]) > iou_threshold {
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

/// Performs per-class NMS.
/// Groups boxes by class_id into a Vec (cheaper than HashMap for small class counts),
/// runs NMS per class, then sorts the merged result by confidence.
#[must_use]
pub fn nms_per_class(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let max_class = boxes.iter().map(|b| b.class_id).max().unwrap_or(0);
    let mut buckets: Vec<Vec<BoundingBox>> = (0..=max_class).map(|_| Vec::new()).collect();

    for &bbox in boxes {
        buckets[bbox.class_id].push(bbox);
    }

    let mut result = Vec::with_capacity(boxes.len());
    for bucket in &buckets {
        if !bucket.is_empty() {
            result.extend(nms(bucket, iou_threshold));
        }
    }

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
