extern crate core;

use crate::model::yolo_type::YoloType;
use crate::session::yolo_session::YoloSession;

pub mod class;
pub mod detection;
pub mod image;
pub mod model;
pub mod session;

// Embed the model at compile time
pub const MODEL_BYTES: &[u8] = include_bytes!("../models/best.onnx");

/// Analyzes an image using the embedded YOLO model.
pub fn analyze_image(
    image_path: &str,
    yolo_type: YoloType,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use the embedded model bytes instead of a file path
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, yolo_type)
        .expect("Failed to create YOLO model from embedded bytes");

    yolo_model
        .process_image(&image_path)
        .expect("Failed to process image");
    Ok(())
}
