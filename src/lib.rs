extern crate core;

use crate::model::yolo_type::YoloType;
use crate::session::yolo_session::YoloSession;

pub mod image;

pub mod session;

pub mod class;
pub mod detection;
pub mod model;

// Embed the model at compile time
pub const MODEL_BYTES: &[u8] = include_bytes!("../models/best.onnx");

/// Analyzes an image using the embedded YOLO model.
pub fn analyze_image(image_path: &str) {
    // Use the embedded model bytes instead of a file path
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, YoloType::YoloV8)
        .expect("Failed to create YOLO model from embedded bytes");

    yolo_model
        .process_image(&image_path)
        .expect("Failed to process image");
}
