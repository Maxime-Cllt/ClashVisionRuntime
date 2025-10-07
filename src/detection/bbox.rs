//! Bounding box utilities and operations.

/// Struct representing a bounding box with coordinates, class ID, and confidence score.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub confidence: f32,
}

impl BoundingBox {
    /// Creates a new bounding box
    #[inline]
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32, class_id: usize, confidence: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            class_id,
            confidence,
        }
    }

    /// Creates a bounding box from center coordinates and dimensions
    #[inline]
    pub fn from_center(
        cx: f32,
        cy: f32,
        width: f32,
        height: f32,
        class_id: usize,
        confidence: f32,
    ) -> Self {
        let half_width = width * 0.5;
        let half_height = height * 0.5;
        Self::new(
            cx - half_width,
            cy - half_height,
            cx + half_width,
            cy + half_height,
            class_id,
            confidence,
        )
    }

    /// Calculates the intersection area with another bounding box
    #[inline]
    #[must_use]
    pub fn intersection(&self, other: &Self) -> f32 {
        let width = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let height = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        width * height
    }

    /// Calculates the union area with another bounding box
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection(other)
    }

    /// Calculates the Intersection over Union (`IoU`) with another bounding box
    #[inline]
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let intersection = self.intersection(other);
        if intersection == 0.0 {
            return 0.0;
        }
        intersection / self.union(other)
    }

    /// Calculates the area of the bounding box
    #[inline]
    #[must_use]
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Returns the center coordinates
    #[inline]
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Returns the width and height
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> (f32, f32) {
        (self.x2 - self.x1, self.y2 - self.y1)
    }

    /// Scales the bounding box coordinates
    #[inline]
    pub fn scale(&mut self, scale_x: f32, scale_y: f32) {
        self.x1 *= scale_x;
        self.x2 *= scale_x;
        self.y1 *= scale_y;
        self.y2 *= scale_y;
    }

    /// Returns a scaled copy of the bounding box
    #[inline]
    pub fn scaled(&self, scale_x: f32, scale_y: f32) -> Self {
        let mut bbox = *self;
        bbox.scale(scale_x, scale_y);
        bbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_creation() {
        let bbox = BoundingBox::new(10.0, 20.0, 50.0, 80.0, 1, 0.9);
        assert_eq!(bbox.area(), 2400.0);
        assert_eq!(bbox.center(), (30.0, 50.0));
        assert_eq!(bbox.dimensions(), (40.0, 60.0));
    }

    #[test]
    fn test_bbox_from_center() {
        let bbox = BoundingBox::from_center(30.0, 50.0, 40.0, 60.0, 1, 0.9);
        assert_eq!(bbox.x1, 10.0);
        assert_eq!(bbox.y1, 20.0);
        assert_eq!(bbox.x2, 50.0);
        assert_eq!(bbox.y2, 80.0);
    }

    #[test]
    fn test_iou() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0, 0, 0.8);
        let iou = bbox1.iou(&bbox2);
        assert!((iou - 0.142_857).abs() < 0.001);
    }
}
