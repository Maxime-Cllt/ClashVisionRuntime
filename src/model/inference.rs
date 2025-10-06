//! Inference logic for different YOLO models

use crate::detection::BoundingBox;
use crate::model::yolov10_inference::Yolov10Inference;
use crate::model::yolov8_inference::Yolov8Inference;
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
pub fn create_inference(model_name: &str) -> Box<dyn YoloInference> {
    match model_name.to_lowercase().as_str() {
        "yolov8" => Box::new(Yolov8Inference),
        "yolov10" => Box::new(Yolov10Inference),
        _ => panic!(
            "Unsupported model: {}. Supported models: yolov8, yolov10",
            model_name
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_model() {
        let result = std::panic::catch_unwind(|| create_inference("unknown_model"));
        assert!(result.is_err());
    }
}
