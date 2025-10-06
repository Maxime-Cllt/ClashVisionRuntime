use std::fmt::Debug;

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

impl TryFrom<&str> for YoloType {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "yolov8" => Ok(YoloType::YoloV8),
            "yolov10" => Ok(YoloType::YoloV10),
            _ => Err(()),
        }
    }
}

impl Debug for YoloType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
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
    
    #[test]
    fn test_yolo_type_try_from() {
        assert_eq!(YoloType::try_from("yolov8").unwrap(), YoloType::YoloV8);
        assert_eq!(YoloType::try_from("YoloV8").unwrap(), YoloType::YoloV8);
        assert_eq!(YoloType::try_from("YOLOV8").unwrap(), YoloType::YoloV8);
        assert_eq!(YoloType::try_from("yolov10").unwrap(), YoloType::YoloV10);
        assert_eq!(YoloType::try_from("YoloV10").unwrap(), YoloType::YoloV10);
        assert_eq!(YoloType::try_from("YOLOV10").unwrap(), YoloType::YoloV10);
        assert!(YoloType::try_from("unknown").is_err());
    }
}
