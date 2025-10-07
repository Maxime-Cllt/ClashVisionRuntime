
mod bbox;
pub mod nms;
pub mod output;
pub mod visualization;

pub use bbox::BoundingBox;

/// Errors that can occur during detection operations
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    #[error("Invalid bounding box coordinates")]
    InvalidBoundingBox,
    #[error("Image processing error: {0}")]
    ImageError(String),
}
