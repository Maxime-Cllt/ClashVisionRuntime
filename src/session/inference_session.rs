use crate::session::SessionError;
use ndarray::{Array4, ArrayD, ArrayViewD};
use tract_onnx::prelude::*;

/// Pure-Rust ONNX inference session backed by [tract](https://github.com/sonos/tract).
///
/// Unlike an `onnxruntime`-backed session, tract is 100% Rust: there is no native
/// `onnxruntime` library to `dlopen` or ship, so the whole crate links into a single
/// static binary that runs anywhere Rust's std runs (notably: no DLL hell on Windows).
#[must_use]
#[non_exhaustive]
pub struct InferenceSession {
    /// Optimized, runnable tract plan. Built once; `run` borrows it immutably.
    model: Arc<TypedRunnableModel>,
}

impl InferenceSession {
    /// Loads and optimizes an ONNX model from raw bytes, fixing the NCHW input shape
    /// to `(1, 3, height, width)` so tract can fully optimize the graph.
    /// Inference is single-threaded: extra threads did not help on the reference model.
    pub fn from_bytes(model_bytes: &[u8], input_size: (u32, u32)) -> Result<Self, SessionError> {
        let (width, height) = input_size;
        let mut reader = std::io::Cursor::new(model_bytes);

        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| SessionError::Inference(format!("failed to read ONNX model: {e}")))?
            .with_input_fact(0, f32::fact([1, 3, height as usize, width as usize]).into())
            .map_err(|e| SessionError::Inference(format!("failed to set input fact: {e}")))?
            .into_optimized()
            .map_err(|e| SessionError::Inference(format!("failed to optimize model: {e}")))?
            .into_runnable()
            .map_err(|e| SessionError::Inference(format!("failed to make model runnable: {e}")))?;

        Ok(Self { model })
    }

    /// Loads and optimizes an ONNX model from a file path.
    pub fn from_path(model_path: &std::path::Path, input_size: (u32, u32)) -> Result<Self, SessionError> {
        let bytes = std::fs::read(model_path)?;
        Self::from_bytes(&bytes, input_size)
    }

    /// Runs inference on a preprocessed NCHW f32 tensor and hands the raw first model
    /// output (e.g. YOLOv8 `output0`, shape `[1, 4 + classes, N]`) to `f` as a borrowed view,
    /// so callers can parse it without copying the output buffer.
    pub fn run_inference_with<R>(
        &self,
        input_image: Array4<f32>,
        f: impl FnOnce(ArrayViewD<'_, f32>) -> R,
    ) -> Result<R, SessionError> {
        // ndarray is the same version as tract's, so the input is moved into the tensor.
        let input_tensor: Tensor = input_image.into();

        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| SessionError::Inference(format!("tract inference failed: {e}")))?;

        let output = outputs
            .first()
            .ok_or_else(|| SessionError::Inference("model produced no output".to_string()))?;

        let view = output
            .to_plain_array_view::<f32>()
            .map_err(|e| SessionError::Inference(format!("failed to read output tensor: {e}")))?;

        Ok(f(view))
    }

    /// Runs inference and returns the raw first model output as an owned dynamic ndarray.
    /// Prefer [`Self::run_inference_with`] on hot paths (this one copies the output).
    pub fn run_inference(&self, input_image: Array4<f32>) -> Result<ArrayD<f32>, SessionError> {
        self.run_inference_with(input_image, |view| view.to_owned())
    }
}
