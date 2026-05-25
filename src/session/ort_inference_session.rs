use ndarray::Array4;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{Session, SessionInputValue, SessionInputs, SessionOutputs};
use ort::value::{Tensor, Value};
use std::borrow::Cow;
use std::path::Path;

/// ONNX Runtime inference session wrapper with optimized configuration.
#[must_use]
#[non_exhaustive]
pub struct OrtInferenceSession {
    session: Session,
}

impl OrtInferenceSession {
    /// Builds a SessionBuilder with platform-tuned execution providers and threading.
    /// - macOS: CoreML (ANE/GPU) with MLProgram + FastPrediction
    /// - Windows/Linux: optimized CPU EP (memory pattern + physical-core threading)
    fn optimized_builder() -> ort::Result<SessionBuilder> {
        let num_threads = num_cpus::get_physical().max(1);

        let mut builder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .with_memory_pattern(true)?;

        // Try CoreML on Apple devices; silently fall back to CPU if unavailable.
        #[cfg(target_vendor = "apple")]
        {
            use ort::ep::CoreML;
            use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};

            let coreml = CoreML::default()
                .with_compute_units(ComputeUnits::All)
                .with_model_format(ModelFormat::MLProgram)
                .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                .with_static_input_shapes(true)
                .build();

            builder = builder.with_execution_providers([coreml])?;
        }

        Ok(builder)
    }

    /// Creates a new ONNX Runtime inference session from the specified model path.
    pub fn new(model_path: &Path) -> ort::Result<Self> {
        let session: Session = Self::optimized_builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Creates a new ONNX Runtime inference session from model bytes.
    pub fn from_bytes(model_bytes: &[u8]) -> ort::Result<Self> {
        let session: Session = Self::optimized_builder()?.commit_from_memory(model_bytes)?;
        Ok(Self { session })
    }

    /// Runs inference on the provided input image tensor.
    /// Takes ownership of the input tensor to enable zero-copy transfer to ONNX Runtime.
    pub fn run_inference(
        &mut self,
        input_image: Array4<f32>,
    ) -> ort::Result<SessionOutputs<'_>> {
        let input_tensor: Tensor<f32> = Tensor::from_array(input_image)?;
        let input_value: SessionInputValue = SessionInputValue::Owned(Value::from(input_tensor));
        let inputs: Vec<(Cow<str>, SessionInputValue)> =
            vec![(Cow::Borrowed("images"), input_value)];

        self.session.run(SessionInputs::from(inputs))
    }
}
