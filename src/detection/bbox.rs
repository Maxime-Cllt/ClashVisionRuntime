//! Bounding box utilities and operations.

use super::DetectionError;

/// Represents a bounding box with coordinates, class ID, and confidence score.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub confidence: f32,
}

impl BoundingBox {
    /// Creates a new bounding box with validation.
    pub fn new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        class_id: usize,
        confidence: f32,
    ) -> Result<Self, DetectionError> {
        if x1 >= x2 || y1 >= y2 || !(0.0..=1.0).contains(&confidence) {
            return Err(DetectionError::InvalidBoundingBox);
        }

        Ok(Self {
            x1,
            y1,
            x2,
            y2,
            class_id,
            confidence,
        })
    }

    /// Calculates the intersection area with another bounding box.
    #[inline]
    pub fn intersection_area(&self, other: &Self) -> f32 {
        let width = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let height = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        width * height
    }

    /// Calculates the area of this bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Calculates the union area with another bounding box.
    #[inline]
    pub fn union_area(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Calculates the Intersection over Union (IoU) with another bounding box.
    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let intersection = self.intersection_area(other);
        if intersection == 0.0 {
            return 0.0;
        }
        intersection / self.union_area(other)
    }

    /// Scales the bounding box coordinates based on image dimensions.
    pub fn scale(&self, from_size: (u32, u32), to_size: (u32, u32)) -> Self {
        let scale_x = to_size.0 as f32 / from_size.0 as f32;
        let scale_y = to_size.1 as f32 / from_size.1 as f32;

        Self {
            x1: self.x1 * scale_x,
            y1: self.y1 * scale_y,
            x2: self.x2 * scale_x,
            y2: self.y2 * scale_y,
            ..*self
        }
    }

    /// Converts to YOLO format (normalized center coordinates and dimensions).
    pub fn to_yolo_format(&self, image_width: u32, image_height: u32) -> YoloBox {
        let x_center = (self.x1 + self.x2) / 2.0 / image_width as f32;
        let y_center = (self.y1 + self.y2) / 2.0 / image_height as f32;
        let width = (self.x2 - self.x1) / image_width as f32;
        let height = (self.y2 - self.y1) / image_height as f32;

        YoloBox {
            class_id: self.class_id,
            x_center,
            y_center,
            width,
            height,
            confidence: self.confidence,
        }
    }
}

/// YOLO format bounding box representation.
#[derive(Debug, Clone, Copy)]
pub struct YoloBox {
    pub class_id: usize, // Class ID of the detected object
    pub x_center: f32,   // Center x-coordinate (normalized)
    pub y_center: f32,   // Center y-coordinate (normalized)
    pub width: f32,      // Width of the bounding box
    pub height: f32,     // Width of the bounding box
    pub confidence: f32, // Optional confidence score
}

impl std::fmt::Display for YoloBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, self.x_center, self.y_center, self.width, self.height
        )
    }
}
