use crate::detection::BoundingBox;
use crate::detection::nms::{nms, nms_per_class};
use crate::detection::output::OutputFormat;
use crate::detection::visualization::DrawConfig;
use crate::image::image_util::load_and_preprocess_f32;
use crate::model::inference::{YoloInferenceImpl, create_inference};
use crate::model::yolo_type::YoloType;
use crate::session::SessionError;
use crate::session::inference_session::InferenceSession;
use crate::session::session_config::SessionConfig;
use image::RgbImage;
use ndarray::Array4;
use std::path::Path;

/// YOLO session struct for managing model inference and image processing
#[must_use]
pub struct YoloSession {
    session: InferenceSession,
    config: SessionConfig,
    inference: YoloInferenceImpl,
}

impl YoloSession {
    /// Creates a new YOLO session with default configuration
    pub fn new(model_path: &str, model_type: YoloType) -> Result<Self, SessionError> {
        Self::with_config(model_path, &model_type, SessionConfig::default())
    }

    /// Creates a new YOLO session with custom configuration
    pub fn with_config(
        model_path: &str,
        model_type: &YoloType,
        config: SessionConfig,
    ) -> Result<Self, SessionError> {
        let session = InferenceSession::from_path(Path::new(model_path), config.input_size)?;
        let inference = create_inference(model_type);

        Ok(Self {
            session,
            config,
            inference,
        })
    }

    /// Creates a new YOLO session with default configuration from model bytes
    pub fn from_bytes(model_bytes: &[u8], model_type: YoloType) -> Result<Self, SessionError> {
        Self::from_bytes_with_config(model_bytes, &model_type, SessionConfig::default())
    }

    /// Creates a new YOLO session with custom configuration from model bytes
    pub fn from_bytes_with_config(
        model_bytes: &[u8],
        model_type: &YoloType,
        config: SessionConfig,
    ) -> Result<Self, SessionError> {
        let session = InferenceSession::from_bytes(model_bytes, config.input_size)?;
        let inference = create_inference(model_type);

        Ok(Self {
            session,
            config,
            inference,
        })
    }

    /// Runs inference on the preprocessed input tensor and returns detected boxes.
    pub fn run_inference(
        &mut self,
        input_tensor: Array4<f32>,
    ) -> Result<Vec<BoundingBox>, SessionError> {
        let output = self.session.run_inference(input_tensor)?;

        // Parse output using appropriate inference implementation
        let boxes = self
            .inference
            .parse_output(output.view(), self.config.confidence_threshold);

        Ok(boxes)
    }

    /// Saves detection outputs
    pub fn save_outputs(
        &self,
        image: &RgbImage,
        boxes: &[BoundingBox],
        image_path: &str,
        output_dir: Option<&str>,
        format: Option<OutputFormat>,
    ) -> Result<(), SessionError> {
        let output_dir_str = output_dir.unwrap_or("output");
        let output_dir = Path::new(output_dir_str);
        let format = format.unwrap_or_default();

        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        let file_name = Path::new(image_path)
            .file_stem()
            .ok_or_else(|| SessionError::ImageProcessing("Invalid image path".to_string()))?;

        let image_output_path = output_dir.join(format!("{}.jpg", file_name.to_string_lossy()));
        let output_path = output_dir.join(format!(
            "{}.{}",
            file_name.to_string_lossy(),
            format.extension()
        ));

        // Save image
        image
            .save(&image_output_path)
            .map_err(|e| SessionError::Io(std::io::Error::other(e)))?;

        // Save YOLO format detections
        OutputFormat::output_detections(boxes, image.dimensions(), &output_path, Some(format))?;

        Ok(())
    }

    /// Processes an image: loads, preprocesses, runs inference, applies NMS, draws boxes, and saves outputs
    pub fn process_image(&mut self, image_path: &str) -> Result<(), SessionError> {
        self.process_image_with_output_dir(image_path, None)
    }

    /// Processes an image with custom output directory.
    /// Uses fused preprocessing pipeline for maximum performance.
    pub fn process_image_with_output_dir(
        &mut self,
        image_path: &str,
        output_dir: Option<&str>,
    ) -> Result<(), SessionError> {
        // Fused pipeline: load + resize + pad + normalize in one go
        let (original_image, normalized) = load_and_preprocess_f32(
            image_path,
            self.config.input_size,
            None,
            None,
        )
        .map_err(|e| SessionError::ImageProcessing(format!("Failed to preprocess image: {e}")))?;

        let mut inferred_boxes = self.run_inference(normalized.image_array)?;

        // Apply NMS if enabled
        if self.config.use_nms {
            inferred_boxes = if self.config.use_per_class_nms {
                nms_per_class(&inferred_boxes, self.config.nms_threshold)
            } else {
                nms(&inferred_boxes, self.config.nms_threshold)
            };
        }

        // Draw boxes — original_image is consumed and boxes are drawn in place
        let result_image = DrawConfig::draw_boxes(
            original_image,
            &inferred_boxes,
            self.config.input_size,
        );

        self.save_outputs(
            &result_image,
            &inferred_boxes,
            image_path,
            output_dir,
            Some(OutputFormat::Json),
        )?;

        Ok(())
    }

    /// Processes multiple images in batch
    pub fn process_images_batch<P: AsRef<Path>>(
        &mut self,
        image_paths: &[P],
        output_dir: Option<&str>,
    ) -> Result<Vec<Result<(), SessionError>>, SessionError> {
        let results = image_paths
            .iter()
            .map(|path| {
                let path_str = path
                    .as_ref()
                    .to_str()
                    .ok_or_else(|| SessionError::ImageProcessing("Invalid path".to_string()))?;
                self.process_image_with_output_dir(path_str, output_dir)
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.input_size, (640, 640));
        assert!(config.use_nms);
        assert_eq!(config.nms_threshold, 0.45);
        assert_eq!(config.confidence_threshold, 0.25);
    }
}
