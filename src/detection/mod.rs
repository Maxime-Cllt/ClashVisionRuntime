pub mod bbox;
pub mod nms;
pub mod output;
mod utils;
pub mod visualization;

pub use bbox::BoundingBox;
pub use nms::non_maximum_suppression;
pub use output::export_to_yolo_format;
pub use visualization::draw_bounding_boxes;

/// Common result type for detection operations
pub type DetectionResult<T> = Result<T, DetectionError>;

/// Errors that can occur during detection operations
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    #[error("Invalid bounding box coordinates")]
    InvalidBoundingBox,
    #[error("Image processing error: {0}")]
    ImageError(String),
    #[error("File I/O error: {0}")]
    IoError(#[from] std::io::Error),
}
