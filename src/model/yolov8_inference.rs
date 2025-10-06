use ndarray::Array;
use crate::detection::BoundingBox;
use crate::model::inference::YoloInference;

/// YOLOv8 inference implementation
pub struct Yolov8Inference;

impl YoloInference for Yolov8Inference {
    
    fn parse_output(&self, output: &Array<f32, ndarray::IxDyn>, confidence_threshold: f32) -> Vec<BoundingBox> {
        let shape = output.shape();
        let reshaped_output = output
            .to_shape((shape[1], shape[2]))
            .expect("Failed to reshape YOLOv8 output");

        let mut boxes = Vec::new();
        let num_detections = reshaped_output.shape()[1];
        let num_classes = reshaped_output.shape()[0] - 4; // Subtract x,y,w,h

        // Optimize by pre-allocating and using iterators
        boxes.reserve(num_detections / 10); // Rough estimate

        for detection_idx in 0..num_detections {
            // Extract coordinates
            let x = reshaped_output[[0, detection_idx]];
            let y = reshaped_output[[1, detection_idx]];
            let w = reshaped_output[[2, detection_idx]];
            let h = reshaped_output[[3, detection_idx]];

            // Find maximum class probability and its index
            let (max_class_id, max_class_prob) = (4..4 + num_classes)
                .map(|class_idx| (class_idx - 4, reshaped_output[[class_idx, detection_idx]]))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));

            if max_class_prob > confidence_threshold {
                let bbox = BoundingBox::from_center(x, y, w, h, max_class_id, max_class_prob);
                boxes.push(bbox);
            }
        }

        boxes
    }
}