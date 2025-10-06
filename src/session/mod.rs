use thiserror::Error;

pub mod ort_inference_session;
pub mod yolo_session;

/// Session-specific errors
#[derive(Error, Debug)]
pub enum SessionError {
   
    #[error("Image processing failed: {0}")]
    ImageProcessing(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
