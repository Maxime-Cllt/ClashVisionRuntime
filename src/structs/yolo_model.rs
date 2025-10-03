use crate::structs::detection::Detection;
use crate::structs::inference_result::InferenceResult;
use ndarray::{Array4, IxDyn};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

/// Represents a YOLO model with its session and class names.
#[must_use]
#[non_exhaustive]
pub struct YoloModel {
    session: ort::Session,
    class_names: Vec<String>,
    _environment: Arc<Environment>, // Keep environment alive
}

impl YoloModel {
    const INPUT_WIDTH: u32 = 640;
    const INPUT_HEIGHT: u32 = 640;
    const CONFIDENCE_THRESHOLD: f32 = 0.5;
    const NMS_THRESHOLD: f32 = 0.4;

    pub fn new(
        model_path: &Path,
        class_names: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create environment first
        let environment = Arc::new(
            Environment::builder()
                .with_name("yolo_inference")
                .build()?
        );

        // Create session with environment reference
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        Ok(Self {
            session,
            class_names,
            _environment: environment, // Store environment to keep it alive
        })
    }

    pub fn preprocess_image(
        &self,
        image_path: &Path,
    ) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let img = image::open(image_path)?.to_rgb8();

        let (orig_width, orig_height) = img.dimensions();

        // Resize image to model input size
        let resized = image::imageops::resize(
            &img,
            Self::INPUT_WIDTH,
            Self::INPUT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        // Convert to ndarray and normalize
        let mut input_array = Array4::<f32>::zeros((
            1,
            3,
            Self::INPUT_HEIGHT as usize,
            Self::INPUT_WIDTH as usize,
        ));

        for (x, y, pixel) in resized.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            input_array[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
            input_array[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
            input_array[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
        }

        Ok(input_array)
    }

    pub fn run_inference(
        &mut self,
        input: Array4<f32>,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        // Convert to CowArray to match ort's expected type
        let input_cow = input.into_dyn().into();
        let input_tensor = ort::Value::from_array(self.session.allocator(), &input_cow)?;

        // Rest of your code...
        let outputs = self.session.run(vec![input_tensor])?;
        let output = &outputs[0];
        let output_view = output.try_extract::<f32>()?;
        let output_array = output_view.view();
        let output_array = output_array.clone().into_dimensionality::<IxDyn>()?;
        let detections = self.parse_detections(output_array)?;
        let filtered_detections = self.apply_nms(detections, Self::NMS_THRESHOLD);

        Ok(filtered_detections)
    }

    fn parse_detections(
        &self,
        output: ndarray::ArrayView<f32, IxDyn>,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let mut detections = Vec::new();

        // YOLO output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
        let shape = output.shape();
        if shape.len() != 3 {
            return Err("Unexpected output shape".into());
        }

        let num_detections = shape[2];
        let num_classes = shape[1] - 4; // subtract bbox coordinates

        for i in 0..num_detections {
            // Extract bbox coordinates (center_x, center_y, width, height)
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            // Convert to x1, y1, x2, y2 format
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Find the class with highest confidence
            let mut max_confidence = 0.0f32;
            let mut best_class_id = 0;

            for class_id in 0..num_classes {
                let confidence = output[[0, 4 + class_id, i]];
                if confidence > max_confidence {
                    max_confidence = confidence;
                    best_class_id = class_id;
                }
            }

            if max_confidence > Self::CONFIDENCE_THRESHOLD {
                detections.push(Detection {
                    class_id: best_class_id,
                    confidence: max_confidence,
                    bbox: [x1, y1, x2, y2],
                });
            }
        }

        Ok(detections)
    }

    fn apply_nms(&self, mut detections: Vec<Detection>, nms_threshold: f32) -> Vec<Detection> {
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppressed[j] || detections[i].class_id != detections[j].class_id {
                    continue;
                }

                let iou = self.calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }

    fn calculate_iou(&self, box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let x1 = box1[0].max(box2[0]);
        let y1 = box1[1].max(box2[1]);
        let x2 = box1[2].min(box2[2]);
        let y2 = box1[3].min(box2[3]);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        let union = area1 + area2 - intersection;

        intersection / union
    }

    pub fn run_inference_on_image(
        &mut self,
        image_path: &Path,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        println!("Processing: {:?}", image_path.file_name().unwrap());

        let input = self.preprocess_image(image_path)?;
        let detections = self.run_inference(input)?;

        Ok(detections)
    }

    pub fn run_inference_on_directory(
        &mut self,
        image_dir: &Path,
        output_dir: &Path,
    ) -> Result<Vec<InferenceResult>, Box<dyn std::error::Error>> {
        let image_extensions = vec!["jpg", "jpeg", "png"];
        fs::create_dir_all(output_dir)?;

        let mut results_summary = Vec::new();

        for entry in fs::read_dir(image_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(extension) = path.extension() {
                if image_extensions
                    .iter()
                    .any(|ext| extension.to_string_lossy().to_lowercase() == *ext)
                {
                    match self.run_inference_on_image(&path) {
                        Ok(detections) => {
                            let image_name =
                                path.file_name().unwrap().to_string_lossy().to_string();

                            // Save annotated image (you can implement this)
                            self.save_annotated_image(&path, &detections, output_dir)?;

                            results_summary.push(InferenceResult {
                                image_name: image_name.clone(),
                                num_detections: detections.len(),
                                detections,
                            });
                        }
                        Err(e) => {
                            eprintln!("Error processing {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        Ok(results_summary)
    }

    pub fn save_annotated_image(
        &self,
        image_path: &Path,
        detections: &[Detection],
        output_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Load original image
        let mut img: RgbImage = image::open(image_path)?.to_rgb8();


        // Loop through detections and annotate
        for detection in detections {
            let class_name = self
                .class_names
                .get(detection.class_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            let x = detection.bbox[0] as i32;
            let y = detection.bbox[1] as i32;
            let w = detection.bbox[2] as i32;
            let h = detection.bbox[3] as i32;


            // Draw bounding box (red)
            let rect = Rect::at(4, 5).of_size(6, 7);
            draw_hollow_rect_mut(&mut img, rect, Rgb([255u8, 0u8, 0u8]));

            // Draw label (white text)
            let scale = Scale { x: 16.0, y: 16.0 };
            let text = format!("{} {:.2}", class_name, detection.confidence);

            // Draw text background (black rectangle)
            let text_width = (text.len() as f32 * scale.x * 0.6) as u32;
            let text_height = scale.y as u32;
            let background_rect = Rect::at(x, y - text_height as i32)
                .of_size(text_width, text_height);
            draw_hollow_rect_mut(&mut img, background_rect, Rgb([0u8, 0u8, 0u8]));
        }

        // Ensure output directory exists
        fs::create_dir_all(output_dir)?;

        // Save annotated image
        let output_path = output_dir.join(format!(
            "annotated_{}",
            image_path.file_name().unwrap().to_string_lossy()
        ));
        img.save(&output_path)?;

        Ok(())
    }
}