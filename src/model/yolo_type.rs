/// Enum representing different types of YOLO models.
#[derive(PartialEq, Eq)]
pub enum YoloType {
    YoloV8,
    YoloV10,
}

impl YoloType {
    /// Returns the string representation of the YoloType variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            YoloType::YoloV8 => "YoloV8",
            YoloType::YoloV10 => "YoloV10",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_type_as_str() {
        assert_eq!(YoloType::YoloV8.as_str(), "YoloV8");
        assert_eq!(YoloType::YoloV10.as_str(), "YoloV10");
    }
}
