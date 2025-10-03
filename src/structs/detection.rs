/// Represents a detected object with its class ID, confidence score, and bounding box coordinates.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
}
