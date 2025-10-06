use ndarray::{ArrayBase, Dim, OwnedRepr};
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionInputValue, SessionInputs, SessionOutputs};
use ort::value::{Tensor, Value};
use std::borrow::Cow;
use std::path::Path;
use std::time::Instant;

/// ONNX Runtime inference session wrapper.
#[must_use]
#[non_exhaustive]
pub struct OrtInferenceSession {
    session: Session,
}

impl OrtInferenceSession {
    /// Creates a new ONNX Runtime inference session from the specified model path.
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session: Session = SessionBuilder::new()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Runs inference on the provided input image tensor.
    pub fn run_inference(
        &mut self,
        input_image: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    ) -> ort::Result<SessionOutputs<'_>> {
        let time_pre_compute = Instant::now();

        let shape: Vec<usize> = input_image.shape().to_vec();
        let raw_data: Vec<f32> = input_image.as_slice().unwrap().to_vec();
        let input_tensor: Tensor<f32> = Tensor::from_array((shape, raw_data.into_boxed_slice()))?;

        let input_value: SessionInputValue = SessionInputValue::Owned(Value::from(input_tensor));
        let inputs: Vec<(Cow<str>, SessionInputValue)> =
            vec![(Cow::Borrowed("images"), input_value)];

        let outputs: SessionOutputs = self.session.run(SessionInputs::from(inputs))?;
        let time_post_compute = Instant::now();

        #[cfg(debug_assertions)]
        {
            println!(
                "Inference time: {:#?}",
                time_post_compute - time_pre_compute
            );
        }

        Ok(outputs)
    }
}
