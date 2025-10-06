//! Non-Maximum Suppression implementation.

use super::bbox::BoundingBox;

/// Configuration for Non-Maximum Suppression.
#[derive(Debug, Clone)]
pub struct NmsConfig {
    pub iou_threshold: f32,
    pub confidence_threshold: f32,
    pub max_detections: Option<usize>,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            confidence_threshold: 0.5,
            max_detections: None,
        }
    }
}

/// Performs Non-Maximum Suppression on a collection of bounding boxes.
pub fn non_maximum_suppression(
    mut boxes: Vec<BoundingBox>,
    config: &NmsConfig,
) -> Vec<BoundingBox> {
    // Filter by confidence threshold
    boxes.retain(|bbox| bbox.confidence >= config.confidence_threshold);

    if boxes.is_empty() {
        return boxes;
    }

    // Sort by confidence in descending order
    boxes.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for (i, current_box) in boxes.iter().enumerate() {
        if suppressed[i] {
            continue;
        }

        selected.push(*current_box);

        // Check if we've reached the maximum number of detections
        if let Some(max_det) = config.max_detections {
            if selected.len() >= max_det {
                break;
            }
        }

        // Suppress overlapping boxes
        for (j, other_box) in boxes.iter().enumerate().skip(i + 1) {
            if !suppressed[j] && current_box.iou(other_box) > config.iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    selected
}

/// Performs class-agnostic NMS (simplified version for backward compatibility).
pub fn nms(boxes: &[BoundingBox], iou_threshold: f32) -> Vec<BoundingBox> {
    let config = NmsConfig {
        iou_threshold,
        confidence_threshold: 0.0,
        max_detections: None,
    };
    non_maximum_suppression(boxes.to_vec(), &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_basic() {
        let boxes = vec![
            BoundingBox::new(10.0, 10.0, 50.0, 50.0, 0, 0.9).unwrap(),
            BoundingBox::new(15.0, 15.0, 55.0, 55.0, 0, 0.8).unwrap(),
            BoundingBox::new(100.0, 100.0, 150.0, 150.0, 1, 0.7).unwrap(),
        ];

        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 2); // Should keep highest confidence from overlapping pair + non-overlapping box
    }
}
