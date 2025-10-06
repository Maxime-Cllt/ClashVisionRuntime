use thiserror::Error;

pub mod ort_inference_session;
mod inference;
pub mod yolo_session;

/// Session-specific errors
#[derive(Error, Debug)]
pub enum SessionError {
    #[error("Model loading failed: {0}")]
    ModelLoad(#[from] ort::Error),

    #[error("Image processing failed: {0}")]
    ImageProcessing(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),
}
