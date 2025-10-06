//! YOLO session management with improved error handling and performance

use crate::detection::BoundingBox;
use crate::detection::nms::{nms, nms_per_class};
use crate::detection::output::output_to_yolo_txt_normalized;
use crate::detection::visualization::{DrawConfig, draw_boxes};
use crate::image::image_util::load_image_u8_default;
use crate::image::image_util::normalize_image_f32;
use crate::image::loaded_image::LoadedImageU8;
use crate::model::inference::{YoloInference, create_inference};
use crate::session::SessionError;
use crate::session::ort_inference_session::OrtInferenceSession;
use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use ort::session::SessionOutputs;
use std::path::Path;

/// Configuration for YOLO session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub input_size: (u32, u32),
    pub use_nms: bool,
    pub nms_threshold: f32,
    pub confidence_threshold: f32,
    pub use_per_class_nms: bool,
    pub draw_config: DrawConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            input_size: (640, 640),
            use_nms: true,
            nms_threshold: 0.45,
            confidence_threshold: 0.25,
            use_per_class_nms: false,
            draw_config: DrawConfig::default(),
        }
    }
}

/// YOLO session struct for managing model inference and image processing
#[must_use]
pub struct YoloSession {
    session: OrtInferenceSession,
    config: SessionConfig,
    model_name: String,
    inference: Box<dyn YoloInference>,
}

impl YoloSession {
    /// Creates a new YOLO session with default configuration
    pub fn new(model_path: &str, model_name: String) -> Result<Self, SessionError> {
        Self::with_config(model_path, model_name, SessionConfig::default())
    }

    /// Creates a new YOLO session with custom configuration
    pub fn with_config(
        model_path: &str,
        model_name: String,
        config: SessionConfig,
    ) -> Result<Self, SessionError> {
        let session = OrtInferenceSession::new(Path::new(model_path))
            .map_err(|e| SessionError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        let inference = create_inference(&model_name);

        Ok(Self {
            session,
            config,
            model_name,
            inference,
        })
    }

    /// Updates session configuration
    pub fn update_config(&mut self, config: SessionConfig) {
        self.config = config;
    }

    /// Runs inference on the preprocessed input tensor
    pub fn run_inference(
        &mut self,
        input_tensor: Array4<f32>,
    ) -> Result<Vec<BoundingBox>, SessionError> {
        let outputs: SessionOutputs = self
            .session
            .run_inference(input_tensor)
            .map_err(|e| SessionError::Inference(e.to_string()))?;

        let (shape, data) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| SessionError::Inference(format!("Failed to extract tensor: {}", e)))?;

        // Convert i64 shape to usize for ndarray
        let shape_usize: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();

        // Build ndarray from ONNX tensor
        let output = ndarray::Array::from_shape_vec(shape_usize, data.to_vec())
            .map_err(|e| SessionError::Inference(format!("Failed to build ndarray: {}", e)))?;

        // Parse output using appropriate inference implementation
        let boxes = self
            .inference
            .parse_output(&output, self.config.confidence_threshold);

        Ok(boxes)
    }

    /// Loads and preprocesses an image
    pub fn load_and_preprocess_image(
        &self,
        image_path: &str,
    ) -> Result<(RgbImage, LoadedImageU8), SessionError> {
        let loaded_image = load_image_u8_default(image_path, self.config.input_size)
            .map_err(|e| SessionError::ImageProcessing(format!("Failed to load image: {}", e)))?;

        let interleaved_data: Vec<u8> = loaded_image
            .image_array
            .view()
            .to_shape((
                3,
                loaded_image.size.height as usize,
                loaded_image.size.width as usize,
            ))
            .map_err(|e| SessionError::ImageProcessing(format!("Failed to reshape image: {}", e)))?
            .permuted_axes((1, 2, 0))
            .iter()
            .copied()
            .collect();

        let img = RgbImage::from_raw(
            loaded_image.size.width,
            loaded_image.size.height,
            interleaved_data,
        )
        .ok_or_else(|| {
            SessionError::ImageProcessing("Failed to create image from raw data".to_string())
        })?;

        Ok((img, loaded_image))
    }

    /// Saves detection outputs
    pub fn save_outputs(
        &self,
        image: &RgbImage,
        boxes: &[BoundingBox],
        image_path: &str,
        output_dir: Option<&str>,
    ) -> Result<(), SessionError> {
        let output_dir_str = output_dir.unwrap_or("output");
        let output_dir = Path::new(output_dir_str);

        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        let file_name = Path::new(image_path)
            .file_stem()
            .ok_or_else(|| SessionError::ImageProcessing("Invalid image path".to_string()))?;

        let image_output_path = output_dir.join(format!("{}.jpg", file_name.to_string_lossy()));
        let txt_output_path = output_dir.join(format!("{}.txt", file_name.to_string_lossy()));

        // Save image
        image
            .save(&image_output_path)
            .map_err(|e| SessionError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        // Save YOLO format detections
        let (image_width, image_height) = image.dimensions();
        output_to_yolo_txt_normalized(
            boxes.to_vec(),
            image_width,
            image_height,
            txt_output_path.to_str().unwrap(),
        )?;

        Ok(())
    }

    /// Processes an image: loads, preprocesses, runs inference, applies NMS, draws boxes, and saves outputs
    pub fn process_image(&mut self, image_path: &str) -> Result<(), SessionError> {
        self.process_image_with_output_dir(image_path, None)
    }

    /// Processes an image with custom output directory
    pub fn process_image_with_output_dir(
        &mut self,
        image_path: &str,
        output_dir: Option<&str>,
    ) -> Result<(), SessionError> {
        let (original_image, loaded_image) = self.load_and_preprocess_image(image_path)?;

        let normalized_image = normalize_image_f32(&loaded_image, None, None);
        let mut inferred_boxes = self.run_inference(normalized_image.image_array)?;

        // Apply NMS if enabled
        if self.config.use_nms {
            inferred_boxes = if self.config.use_per_class_nms {
                nms_per_class(&inferred_boxes, self.config.nms_threshold)
            } else {
                nms(&inferred_boxes, self.config.nms_threshold)
            };
        }

        // Draw boxes with custom configuration
        let result_image = draw_boxes(
            &DynamicImage::ImageRgb8(original_image),
            &inferred_boxes,
            self.config.input_size,
        );

        self.save_outputs(&result_image, &inferred_boxes, image_path, output_dir)?;

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

    /// Returns inference statistics
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_name: self.model_name.clone(),
            input_size: self.config.input_size,
            confidence_threshold: self.config.confidence_threshold,
            nms_threshold: self.config.nms_threshold,
            use_nms: self.config.use_nms,
        }
    }
}

/// Model information struct
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_name: String,
    pub input_size: (u32, u32),
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub use_nms: bool,
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
