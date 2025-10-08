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