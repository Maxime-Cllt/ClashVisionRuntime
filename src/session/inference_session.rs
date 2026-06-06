use crate::session::SessionError;
use ndarray::{Array4, ArrayD, IxDyn};
use std::path::Path;
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
    model: TypedRunnableModel<TypedModel>,
}

impl InferenceSession {
    /// Loads and optimizes an ONNX model from raw bytes, fixing the NCHW input shape
    /// to `(1, 3, height, width)` so tract can fully optimize the graph.
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
    pub fn from_path(model_path: &Path, input_size: (u32, u32)) -> Result<Self, SessionError> {
        let bytes = std::fs::read(model_path)?;
        Self::from_bytes(&bytes, input_size)
    }

    /// Runs inference on a preprocessed NCHW f32 tensor and returns the raw first model
    /// output as a dynamic ndarray (e.g. YOLOv8 `output0`, shape `[1, 4 + classes, N]`).
    pub fn run_inference(&self, input_image: Array4<f32>) -> Result<ArrayD<f32>, SessionError> {
        // tract carries its own ndarray version, so we bridge via raw contiguous slices
        // (input) and shape+slice (output) rather than passing ndarray types across the boundary.
        let standard = input_image.as_standard_layout();
        let shape = standard.shape().to_vec();
        let slice: &[f32] = standard
            .as_slice()
            .ok_or_else(|| SessionError::Inference("input tensor is not contiguous".to_string()))?;

        let input_tensor = Tensor::from_shape(&shape, slice)
            .map_err(|e| SessionError::Inference(format!("failed to build input tensor: {e}")))?;

        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| SessionError::Inference(format!("tract inference failed: {e}")))?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| SessionError::Inference("model produced no output".to_string()))?;

        let out_shape: Vec<usize> = output.shape().to_vec();
        let out_slice: &[f32] = output
            .as_slice::<f32>()
            .map_err(|e| SessionError::Inference(format!("failed to read output tensor: {e}")))?;

        ArrayD::from_shape_vec(IxDyn(&out_shape), out_slice.to_vec())
            .map_err(|e| SessionError::Inference(format!("failed to build output array: {e}")))
    }
}
