use crate::structs::detection::Detection;

/// Represents the result of an inference operation, including the image name,
#[derive(Debug)]
#[must_use]
#[non_exhaustive]
pub struct InferenceResult {
    pub image_name: String,
    pub detections: Vec<Detection>,
    pub num_detections: usize,
}
