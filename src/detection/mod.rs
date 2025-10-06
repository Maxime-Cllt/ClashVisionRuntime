pub mod bbox;
pub mod nms;
pub mod output;
mod utils;
pub mod visualization;

pub use bbox::BoundingBox;
pub use visualization::draw_bounding_boxes;

/// Errors that can occur during detection operations
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    #[error("Invalid bounding box coordinates")]
    InvalidBoundingBox,
    #[error("Image processing error: {0}")]
    ImageError(String),
}
