use crate::structs::detection::Detection;
use tch::{CModule, Device, Kind, Tensor};
use image::{GenericImageView, RgbImage};

#[must_use]
#[non_exhaustive]
pub struct YOLOModel {
    model: CModule,
    device: Device,
}

impl YOLOModel {
    const INPUT_SIZE: i64 = 640; // YOLOv8 default input size
    const CONFIDENCE_THRESHOLD: f32 = 0.25;
    const NMS_THRESHOLD: f32 = 0.45;

    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let model = CModule::load_on_device(model_path, device)?;
        Ok(YOLOModel { model, device })
    }

    /// Letterbox resize with padding (YOLO style)
    fn preprocess_image(&self, image_path: &str) -> anyhow::Result<(Tensor, f32, f32, u32, u32)> {
        let img = image::open(image_path)?.to_rgb8();
        let (orig_w, orig_h) = img.dimensions();

        // Scale ratio
        let r = (Self::INPUT_SIZE as f32 / orig_w as f32)
            .min(Self::INPUT_SIZE as f32 / orig_h as f32);

        let new_w = (orig_w as f32 * r).round() as u32;
        let new_h = (orig_h as f32 * r).round() as u32;

        // Resize with aspect ratio
        let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);

        // Create 640x640 canvas and paste resized image (pad on right/bottom)
        let mut canvas = RgbImage::from_pixel(Self::INPUT_SIZE as u32, Self::INPUT_SIZE as u32, image::Rgb([114, 114, 114]));
        image::imageops::replace(&mut canvas, &resized, 0, 0);

        // To Tensor
        let mut data = Vec::with_capacity((Self::INPUT_SIZE * Self::INPUT_SIZE * 3) as usize);
        for p in canvas.pixels() {
            data.push(p[0] as f32 / 255.0);
            data.push(p[1] as f32 / 255.0);
            data.push(p[2] as f32 / 255.0);
        }

        let tensor = Tensor::from_slice(&data)
            .view([1, Self::INPUT_SIZE, Self::INPUT_SIZE, 3])
            .permute([0, 3, 1, 2]) // NCHW
            .to_device(self.device)
            .to_kind(Kind::Float);

        Ok((tensor, r, r, orig_w, orig_h))
    }

    fn postprocess_output(
        &self,
        output: Tensor,
        gain: f32,
        pad_x: u32,
        pad_y: u32,
        orig_w: u32,
        orig_h: u32,
    ) -> anyhow::Result<Vec<Detection>> {
        // Expected shape: [batch, num_preds, 4+num_classes]
        let output = output.squeeze_dim(0); // [num_preds, attrs]
        let num_preds = output.size()[0];
        let num_attrs = output.size()[1];

        let mut detections = Vec::new();

        for i in 0..num_preds {
            let cx = output.double_value(&[i, 0]) as f32;
            let cy = output.double_value(&[i, 1]) as f32;
            let w  = output.double_value(&[i, 2]) as f32;
            let h  = output.double_value(&[i, 3]) as f32;

            // best class
            let mut best_score = 0.0;
            let mut best_class = 0;
            for j in 4..num_attrs {
                let score = output.double_value(&[i, j]) as f32;
                if score > best_score {
                    best_score = score;
                    best_class = j - 4;
                }
            }

            if best_score > Self::CONFIDENCE_THRESHOLD {
                // Convert cx,cy,w,h -> x1,y1,x2,y2
                let mut x1 = cx - w / 2.0;
                let mut y1 = cy - h / 2.0;
                let mut x2 = cx + w / 2.0;
                let mut y2 = cy + h / 2.0;

                // Undo letterbox scaling
                x1 = (x1 / gain).clamp(0.0, orig_w as f32);
                y1 = (y1 / gain).clamp(0.0, orig_h as f32);
                x2 = (x2 / gain).clamp(0.0, orig_w as f32);
                y2 = (y2 / gain).clamp(0.0, orig_h as f32);

                detections.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    confidence: best_score,
                    class_id: best_class as usize,
                });
            }
        }

        Ok(self.apply_nms(detections)?)
    }

    fn apply_nms(&self, mut dets: Vec<Detection>) -> anyhow::Result<Vec<Detection>> {
        dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut keep = Vec::new();

        while let Some(current) = dets.pop() {
            keep.push(current.clone());
            dets.retain(|d| {
                if d.class_id != current.class_id {
                    true
                } else {
                    let iou = self.calculate_iou(&current.bbox, &d.bbox);
                    iou < Self::NMS_THRESHOLD
                }
            });
        }
        Ok(keep)
    }

    fn calculate_iou(&self, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        let x1 = a[0].max(b[0]);
        let y1 = a[1].max(b[1]);
        let x2 = a[2].min(b[2]);
        let y2 = a[3].min(b[3]);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }
        let inter = (x2 - x1) * (y2 - y1);
        let area1 = (a[2] - a[0]) * (a[3] - a[1]);
        let area2 = (b[2] - b[0]) * (b[3] - b[1]);
        inter / (area1 + area2 - inter)
    }

    pub fn inference(&self, image_path: &str) -> anyhow::Result<Vec<Detection>> {
        let (input, gain, _, orig_w, orig_h) = self.preprocess_image(image_path)?;
        let output = self.model.forward_ts(&[input])?;
        let dets = self.postprocess_output(output, gain, 0, 0, orig_w, orig_h)?;
        Ok(dets)
    }
}
