//! Inference logic for different YOLO models

use crate::detection::BoundingBox;
use crate::model::yolo_type::YoloType;
use crate::model::yolov8_inference::Yolov8Inference;
use crate::model::yolov10_inference::Yolov10Inference;
use ndarray::ArrayViewD;

/// Trait for YOLO model inference
pub trait YoloInference {
    /// Parses the model output to extract bounding boxes
    fn parse_output(
        &self,
        output: ArrayViewD<'_, f32>,
        confidence_threshold: f32,
    ) -> Vec<BoundingBox>;
}

/// Concrete enum replacing `Box<dyn YoloInference>` to eliminate vtable overhead
/// and enable compiler inlining of the hot parse_output path.
pub enum YoloInferenceImpl {
    V8(Yolov8Inference),
    V10(Yolov10Inference),
}

impl YoloInferenceImpl {
    /// Parses the model output using the appropriate inference implementation.
    #[inline]
    pub fn parse_output(
        &self,
        output: ArrayViewD<'_, f32>,
        confidence_threshold: f32,
    ) -> Vec<BoundingBox> {
        match self {
            Self::V8(inner) => inner.parse_output(output, confidence_threshold),
            Self::V10(inner) => inner.parse_output(output, confidence_threshold),
        }
    }
}

/// Factory function to create appropriate inference implementation
#[must_use]
pub fn create_inference(model_name: &YoloType) -> YoloInferenceImpl {
    match model_name {
        YoloType::YoloV8 => YoloInferenceImpl::V8(Yolov8Inference),
        YoloType::YoloV10 => YoloInferenceImpl::V10(Yolov10Inference),
    }
}
