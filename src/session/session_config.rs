use crate::detection::visualization::DrawConfig;

/// Configuration for YOLO session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub input_size: (u32, u32),
    pub use_nms: bool,
    pub nms_threshold: f32,
    pub confidence_threshold: f32,
    pub use_per_class_nms: bool,
    pub draw_config: DrawConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            input_size: (640, 640),
            use_nms: true,
            nms_threshold: 0.45,
            confidence_threshold: 0.25,
            use_per_class_nms: false,
            draw_config: DrawConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.input_size, (640, 640));
        assert!(config.use_nms);
        assert_eq!(config.nms_threshold, 0.45);
        assert_eq!(config.confidence_threshold, 0.25);
        assert!(!config.use_per_class_nms);
        assert_eq!(config.draw_config, DrawConfig::default());
    }

    #[test]
    fn test_session_config_custom() {
        let config = SessionConfig {
            input_size: (800, 600),
            use_nms: false,
            nms_threshold: 0.5,
            confidence_threshold: 0.3,
            use_per_class_nms: true,
            draw_config: DrawConfig {
                line_width: 0.0,
                alpha_blend: false,
                show_confidence: false,
                font_size: 0.0,
            },
        };
        assert_eq!(config.input_size, (800, 600));
        assert!(!config.use_nms);
        assert_eq!(config.nms_threshold, 0.5);
        assert_eq!(config.confidence_threshold, 0.3);
        assert!(config.use_per_class_nms);
    }
}
