//! Inference logic for different YOLO models

use crate::detection::BoundingBox;
use crate::model::yolo_type::YoloType;
use crate::model::yolov8_inference::Yolov8Inference;
use crate::model::yolov10_inference::Yolov10Inference;
use ndarray::Array;

/// Trait for YOLO model inference
pub trait YoloInference {
    /// Parses the model output to extract bounding boxes
    fn parse_output(
        &self,
        output: &Array<f32, ndarray::IxDyn>,
        confidence_threshold: f32,
    ) -> Vec<BoundingBox>;
}

/// Factory function to create appropriate inference implementation
#[must_use]
pub fn create_inference(model_name: &YoloType) -> Box<dyn YoloInference> {
    match model_name {
        YoloType::YoloV8 => Box::new(Yolov8Inference),
        YoloType::YoloV10 => Box::new(Yolov10Inference),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_yolov8_parse_output() {
        let inference = Yolov8Inference;
        let output = array![
            [0.1, 0.1, 0.4, 0.4, 0.9, 0.1], // High confidence box
            [0.5, 0.5, 0.7, 0.7, 0.4, 0.2], // Low confidence box
        ]
        .into_dyn();
        let boxes = inference.parse_output(&output, 0.5);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].confidence, 0.9);
    }
}
