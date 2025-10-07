#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    /// Creates a new `ImageSize`
    #[inline]
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Calculates the aspect ratio (width / height)
    pub fn aspect_ratio(self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_size_creation() {
        let size = ImageSize::new(640, 480);
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
        assert!((size.aspect_ratio() - 4.0 / 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aspect_ratio() {
        let size = ImageSize::new(1920, 1080);
        assert!((size.aspect_ratio() - (16.0 / 9.0)).abs() < f32::EPSILON);
    }
}
