//! Output utilities for saving detection results

use super::bbox::BoundingBox;
use serde::Serialize;
use std::fs;
use std::io::{self};
use std::path::Path;

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputFormat {
    #[default]
    Yolo,
    Json,
}

impl Serialize for OutputFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            Self::Yolo => "yolo",
            Self::Json => "json",
        };
        serializer.serialize_str(s)
    }
}

impl OutputFormat {
    /// Outputs detection results in different formats
    pub fn output_detections(
        boxes: &[BoundingBox],
        image_dimensions: (u32, u32),
        output_path: &Path,
        format: Option<Self>,
    ) -> io::Result<()> {
        let format: Self = format.unwrap_or_default();
        match format {
            Self::Yolo => Self::output_to_yolo_txt_normalized(
                boxes,
                image_dimensions.0,
                image_dimensions.1,
                output_path.to_str().unwrap(),
            ),
            Self::Json => Self::output_to_coco_json(boxes, image_dimensions, output_path),
        }
    }

    /// Outputs in COCO JSON format to a json file
    fn output_to_coco_json(
        boxes: &[BoundingBox],
        image_dimensions: (u32, u32),
        output_path: &Path,
    ) -> io::Result<()> {
        let stub = serde_json::json!({
            "images": [{
                "width": image_dimensions.0,
                "height": image_dimensions.1,
                "file_name": output_path.file_stem().unwrap().to_str().unwrap()
            }],
            "detections": [],
        });

        // Loop through boxes and add to detections
        let mut detections = Vec::new();
        for (i, bbox) in boxes.iter().enumerate() {
            let (width, height) = bbox.dimensions();
            detections.push(serde_json::json!({
                "id": i + 1,
                "category_id": bbox.class_id,
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
                "width": width,
                "height": height,
                "score": bbox.confidence,
            }));
        }
        let mut output = stub;
        output["detections"] = serde_json::Value::Array(detections);
        fs::write(output_path, serde_json::to_string_pretty(&output).unwrap())?;

        Ok(())
    }

    /// Outputs normalized YOLO format with error handling
    fn output_to_yolo_txt_normalized(
        boxes: &[BoundingBox],
        image_width: u32,
        image_height: u32,
        output_path: &str,
    ) -> io::Result<()> {
        if boxes.is_empty() {
            return fs::write(output_path, "");
        }

        let img_width_f = image_width as f32;
        let img_height_f = image_height as f32;

        // Pre-allocate string with estimated capacity
        let estimated_size = boxes.len() * 50; // Rough estimate: 50 chars per line
        let mut yolo_output = String::with_capacity(estimated_size);

        for bbox in boxes {
            let (center_x, center_y) = bbox.center();
            let (width, height) = bbox.dimensions();

            // Normalize coordinates
            let norm_center_x = center_x / img_width_f;
            let norm_center_y = center_y / img_height_f;
            let norm_width = width / img_width_f;
            let norm_height = height / img_height_f;

            // Format with appropriate precision
            yolo_output.push_str(&format!(
                "{} {:.6} {:.6} {:.6} {:.6}\n",
                bbox.class_id, norm_center_x, norm_center_y, norm_width, norm_height
            ));
        }

        fs::write(output_path, yolo_output)
    }

    /// Returns the file extension for the output format
    #[inline]
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Yolo => "txt",
            Self::Json => "json",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::clash_class::ClashClass;
    use tempfile::NamedTempFile;

    #[test]
    fn test_yolo_output_yolo_format() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let boxes = vec![
            BoundingBox::new(10.0, 20.0, 50.0, 80.0, 1, 0.9),
            BoundingBox::new(30.0, 40.0, 70.0, 90.0, 2, 0.8),
        ];

        OutputFormat::output_to_yolo_txt_normalized(
            &boxes,
            100,
            100,
            temp_file.path().to_str().unwrap(),
        )?;

        let content = fs::read_to_string(temp_file.path())?;
        let expected =
            "1 0.300000 0.500000 0.400000 0.600000\n2 0.500000 0.650000 0.400000 0.500000\n";
        assert_eq!(content, expected);

        Ok(())
    }

    #[test]
    fn test_yolo_output_single_box() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let boxes = vec![BoundingBox::new(
            10.0,
            20.0,
            50.0,
            80.0,
            ClashClass::GoldStorage.into(),
            1.0,
        )];

        OutputFormat::output_to_yolo_txt_normalized(
            &boxes,
            100,
            100,
            temp_file.path().to_str().unwrap(),
        )?;

        let content = fs::read_to_string(temp_file.path())?;
        assert_eq!(content.trim(), "1 0.300000 0.500000 0.400000 0.600000");

        Ok(())
    }

    #[test]
    fn test_yolo_output_json() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let boxes = vec![BoundingBox::new(
            10.0,
            20.0,
            50.0,
            80.0,
            ClashClass::GoldStorage.into(),
            1.0,
        )];

        OutputFormat::output_to_coco_json(&boxes, (100, 100), temp_file.path())?;

        let content = fs::read_to_string(temp_file.path())?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        assert_eq!(json["images"][0]["width"], 100);
        assert_eq!(json["images"][0]["height"], 100);
        assert_eq!(
            json["detections"][0]["category_id"],
            ClashClass::GoldStorage as usize
        );
        assert_eq!(json["detections"][0]["score"], 1.0);
        Ok(())
    }

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Yolo.extension(), "txt");
        assert_eq!(OutputFormat::Json.extension(), "json");
    }
}
