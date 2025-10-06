//! Inference logic for different YOLO models

use crate::detection::BoundingBox;
use crate::model::yolo_type::YoloType;
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
    match YoloType::try_from(model_name) {
        Ok(YoloType::YoloV8) => Box::new(Yolov8Inference),
        Ok(YoloType::YoloV10) => Box::new(Yolov10Inference),
        Err(()) => panic!("Unsupported model: {model_name}. Supported models: yolov8, yolov10"),
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
