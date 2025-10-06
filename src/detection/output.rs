//! Output utilities for exporting detection results.

use super::bbox::{BoundingBox, YoloBox};
use super::{DetectionError, DetectionResult};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Export configuration for different output formats.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub include_confidence: bool,
    pub precision: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_confidence: false,
            precision: 6,
        }
    }
}

/// Exports bounding boxes to YOLO format text file.
pub fn export_to_yolo_format<P: AsRef<Path>>(
    boxes: &[BoundingBox],
    image_dimensions: (u32, u32),
    output_path: P,
    config: Option<ExportConfig>,
) -> DetectionResult<()> {
    let config = config.unwrap_or_default();
    let (image_width, image_height) = image_dimensions;

    if boxes.is_empty() {
        return write_empty_file(output_path);
    }

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    for bbox in boxes {
        let yolo_box = bbox.to_yolo_format(image_width, image_height);
        let line = format_yolo_line(&yolo_box, &config);
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    Ok(())
}

/// Formats a single YOLO box line according to the configuration.
fn format_yolo_line(yolo_box: &YoloBox, config: &ExportConfig) -> String {
    let precision = config.precision;

    if config.include_confidence {
        format!(
            "{} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$}",
            yolo_box.class_id,
            yolo_box.x_center,
            yolo_box.y_center,
            yolo_box.width,
            yolo_box.height,
            yolo_box.confidence,
            prec = precision
        )
    } else {
        format!(
            "{} {:.prec$} {:.prec$} {:.prec$} {:.prec$}",
            yolo_box.class_id,
            yolo_box.x_center,
            yolo_box.y_center,
            yolo_box.width,
            yolo_box.height,
            prec = precision
        )
    }
}

/// Writes an empty file (for cases with no detections).
fn write_empty_file<P: AsRef<Path>>(output_path: P) -> DetectionResult<()> {
    File::create(output_path)?;
    Ok(())
}

/// Exports detection results to JSON format.
pub fn export_to_json<P: AsRef<Path>>(
    boxes: &[BoundingBox],
    image_dimensions: (u32, u32),
    output_path: P,
) -> DetectionResult<()> {
    use serde_json;

    #[derive(serde::Serialize)]
    struct JsonDetection {
        class_id: usize,
        confidence: f32,
        bbox: [f32; 4],            // [x1, y1, x2, y2]
        normalized_bbox: [f32; 4], // [x1, y1, x2, y2] normalized to [0, 1]
    }

    let (width, height) = image_dimensions;
    let detections: Vec<JsonDetection> = boxes
        .iter()
        .map(|bbox| JsonDetection {
            class_id: bbox.class_id,
            confidence: bbox.confidence,
            bbox: [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            normalized_bbox: [
                bbox.x1 / width as f32,
                bbox.y1 / height as f32,
                bbox.x2 / width as f32,
                bbox.y2 / height as f32,
            ],
        })
        .collect();

    let json_output = serde_json::to_string_pretty(&detections).map_err(|e| {
        DetectionError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("JSON serialization error: {}", e),
        ))
    })?;

    std::fs::write(output_path, json_output)?;
    Ok(())
}

// Backward compatibility function
pub fn output_to_yolo_txt(
    boxes: Vec<BoundingBox>,
    image_width: u32,
    image_height: u32,
    output_path: &str,
) {
    if let Err(e) = export_to_yolo_format(&boxes, (image_width, image_height), output_path, None) {
        eprintln!("Failed to write YOLO output: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_yolo_export() -> DetectionResult<()> {
        let boxes = vec![
            BoundingBox::new(10.0, 20.0, 30.0, 40.0, 0, 0.9)?,
            BoundingBox::new(50.0, 60.0, 70.0, 80.0, 1, 0.8)?,
        ];

        let temp_file = NamedTempFile::new()?;
        export_to_yolo_format(&boxes, (100, 100), temp_file.path(), None)?;

        let content = std::fs::read_to_string(temp_file.path())?;
        let lines: Vec<&str> = content.trim().split('\n').collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("0 "));
        assert!(lines[1].starts_with("1 "));

        Ok(())
    }
}
