use crate::structs::detection::Detection;
use tch::{vision, Device, Tensor};

/// Represents a YOLO model with its session and class names.
#[must_use]
#[non_exhaustive]
pub struct YOLOModel {
    model: tch::CModule,
    device: Device,
}

impl YOLOModel {

    const INPUT_SIZE: i64 = 640; // Standard YOLOv8 input size
    const CONFIDENCE_THRESHOLD: f64 = 0.5;
    const NMS_THRESHOLD: f64 = 0.4;

    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        // Use CPU device for cross-platform compatibility
        let device = Device::Cpu;

        // Load the PyTorch model
        let model = tch::CModule::load_on_device(model_path, device)?;

        Ok(YOLOModel { model, device })
    }

    fn preprocess_image(&self, image_path: &str) -> anyhow::Result<(Tensor, f32, f32)> {
        // Load image
        let img = image::open(image_path)?;
        let img = img.to_rgb8();
        let (orig_width, orig_height) = img.dimensions();

        // Calculate scale factors
        let scale_x = Self::INPUT_SIZE as f32 / orig_width as f32;
        let scale_y = Self::INPUT_SIZE as f32 / orig_height as f32;

        // Resize image to model input size
        let resized_img = image::imageops::resize(
            &img,
            Self::INPUT_SIZE as u32,
            Self::INPUT_SIZE as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert image to tensor and normalize
        let mut tensor_data = Vec::new();
        for pixel in resized_img.pixels() {
            tensor_data.push(pixel[0] as f32 / 255.0); // R
            tensor_data.push(pixel[1] as f32 / 255.0); // G
            tensor_data.push(pixel[2] as f32 / 255.0); // B
        }

        let tensor = Tensor::from_slice(&tensor_data)
            .view([1, 3, Self::INPUT_SIZE, Self::INPUT_SIZE])
            .to_device(self.device);

        Ok((tensor, scale_x, scale_y))
    }

    fn postprocess_output(
        &self,
        output: Tensor,
        scale_x: f32,
        scale_y: f32,
    ) -> anyhow::Result<Vec<Detection>> {
        let output = output.squeeze_dim(0); // Remove batch dimension

        // YOLOv8 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
        // For 2 classes: [batch, 6, 8400] where 6 = 4 (bbox) + 2 (classes)
        let output = output.transpose(0, 1); // [8400, 6]

        let mut output_array : Vec<f32> = Vec::new();

        for i in 0..output.size()[0] {
            for j in 0..output.size()[1] {
                output_array.push(output.double_value(&[i, j]) as f32);
            }
        }


        let num_detections = output_array.len() / 6; // 6 = 4 bbox + 2 classes
        let mut detections = Vec::new();

        for i in 0..num_detections {
            let base_idx = i * 6;

            // Extract bbox coordinates (center_x, center_y, width, height)
            let cx = output_array[base_idx];
            let cy = output_array[base_idx + 1];
            let w = output_array[base_idx + 2];
            let h = output_array[base_idx + 3];

            // Extract class scores
            let class_0_score = output_array[base_idx + 4];
            let class_1_score = output_array[base_idx + 5];

            // Find the class with highest confidence
            let (class_id, confidence) = if class_0_score > class_1_score {
                (0, class_0_score)
            } else {
                (1, class_1_score)
            };

            // Filter by confidence threshold
            if confidence > Self::CONFIDENCE_THRESHOLD as f32 {
                // Convert from center format to corner format
                let x1 = (cx - w / 2.0) / scale_x;
                let y1 = (cy - h / 2.0) / scale_y;
                let x2 = (cx + w / 2.0) / scale_x;
                let y2 = (cy + h / 2.0) / scale_y;

                detections.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    confidence,
                    class_id,
                });
            }
        }

        // Apply Non-Maximum Suppression
        let filtered_detections = self.apply_nms(detections)?;

        Ok(filtered_detections)
    }

    fn apply_nms(&self, mut detections: Vec<Detection>) -> anyhow::Result<Vec<Detection>> {
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();

        while !detections.is_empty() {
            let current = detections.remove(0);
            keep.push(current.clone());

            detections.retain(|det| {
                if det.class_id != current.class_id {
                    true // Keep detections of different classes
                } else {
                    let iou = self.calculate_iou(&current.bbox, &det.bbox);
                    iou < Self::NMS_THRESHOLD as f32
                }
            });
        }

        Ok(keep)
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

    pub fn inference(&self, image_path: &str) -> anyhow::Result<Vec<Detection>> {
        let (input_tensor, scale_x, scale_y) = self.preprocess_image(image_path)?;

        // Run inference
        let output = self.model.forward_ts(&[input_tensor])?;

        // Post-process results
        let detections = self.postprocess_output(output, scale_x, scale_y)?;

        Ok(detections)
    }
}