//! Output utilities for saving detection results

use super::bbox::BoundingBox;
use std::fs;
use std::io::{self};
use std::path::Path;

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Yolo,
    Coco,
    Pascal,
}

/// Outputs bounding boxes to a YOLO-format text file
pub fn output_to_yolo_txt(
    boxes: Vec<BoundingBox>,
    image_width: u32,
    image_height: u32,
    output_path: &str,
) -> io::Result<()> {
    output_to_yolo_txt_normalized(boxes, image_width, image_height, output_path)
}

/// Outputs normalized YOLO format with error handling
pub fn output_to_yolo_txt_normalized(
    boxes: Vec<BoundingBox>,
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

    for bbox in &boxes {
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

/// Outputs detection results in different formats
pub fn output_detections(
    boxes: &[BoundingBox],
    image_dimensions: (u32, u32),
    output_path: &Path,
    format: OutputFormat,
) -> io::Result<()> {
    match format {
        OutputFormat::Yolo => output_to_yolo_txt_normalized(
            boxes.to_vec(),
            image_dimensions.0,
            image_dimensions.1,
            output_path.to_str().unwrap(),
        ),
        OutputFormat::Coco => output_to_coco_json(boxes, image_dimensions, output_path),
        OutputFormat::Pascal => output_to_pascal_xml(boxes, image_dimensions, output_path),
    }
}

/// Placeholder for COCO JSON output
fn output_to_coco_json(
    _boxes: &[BoundingBox],
    _image_dimensions: (u32, u32),
    _output_path: &Path,
) -> io::Result<()> {
    // Implementation for COCO format would go here
    todo!("COCO JSON output not implemented yet")
}

/// Placeholder for Pascal VOC XML output
fn output_to_pascal_xml(
    _boxes: &[BoundingBox],
    _image_dimensions: (u32, u32),
    _output_path: &Path,
) -> io::Result<()> {
    // Implementation for Pascal VOC format would go here
    todo!("Pascal VOC XML output not implemented yet")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_yolo_output_empty() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let boxes = vec![];

        output_to_yolo_txt_normalized(boxes, 640, 480, temp_file.path().to_str().unwrap())?;

        let content = fs::read_to_string(temp_file.path())?;
        assert!(content.is_empty());
        Ok(())
    }

    #[test]
    fn test_yolo_output_single_box() -> io::Result<()> {
        let temp_file = NamedTempFile::new()?;
        let boxes = vec![BoundingBox::new(10.0, 20.0, 50.0, 80.0, 1, 0.9)];

        output_to_yolo_txt_normalized(boxes, 100, 100, temp_file.path().to_str().unwrap())?;

        let content = fs::read_to_string(temp_file.path())?;
        assert!(content.contains("1 0.300000 0.500000 0.400000 0.600000"));
        Ok(())
    }
}
