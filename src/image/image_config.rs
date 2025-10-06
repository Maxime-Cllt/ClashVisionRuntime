use crate::image::image_size::ImageSize;
use crate::image::norm_config::NormalizationConfig;
use crate::image::{IMAGENET_MEAN, IMAGENET_STD, PADDING_COLOR};
use image::imageops::FilterType;

/// Configuration for image loading and preprocessing
#[derive(Debug, Clone)]
pub struct ImageConfig {
    pub target_size: ImageSize,
    pub filter_type: FilterType,
    pub padding_color: [u8; 3],
    pub normalization: NormalizationConfig,
}

impl ImageConfig {
    /// Creates a new ImageConfig with specified parameters
    pub fn new(
        target_size: ImageSize,
        filter_type: FilterType,
        padding_color: [u8; 3],
        normalization: NormalizationConfig,
    ) -> Self {
        Self {
            target_size,
            filter_type,
            padding_color,
            normalization,
        }
    }
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            target_size: ImageSize::new(224, 224),
            filter_type: FilterType::Lanczos3, // Better quality than Nearest
            padding_color: PADDING_COLOR,
            normalization: NormalizationConfig {
                mean: IMAGENET_MEAN,
                std: IMAGENET_STD,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::FilterType;

    #[test]
    fn test_default_image_config() {
        let config = ImageConfig::default();
        assert_eq!(config.target_size.width, 224);
        assert_eq!(config.target_size.height, 224);
        assert_eq!(config.filter_type, FilterType::Lanczos3);
        assert_eq!(config.padding_color, PADDING_COLOR);
        assert_eq!(config.normalization.mean, IMAGENET_MEAN);
        assert_eq!(config.normalization.std, IMAGENET_STD);
    }

    #[test]
    fn test_custom_image_config() {
        let custom_size = ImageSize::new(300, 300);
        let custom_filter = FilterType::Nearest;
        let custom_padding = [255, 0, 0]; // Red padding
        let custom_norm = NormalizationConfig {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        };
        let config = ImageConfig {
            target_size: custom_size,
            filter_type: custom_filter,
            padding_color: custom_padding,
            normalization: custom_norm.clone(),
        };
        assert_eq!(config.target_size, custom_size);
        assert_eq!(config.filter_type, custom_filter);
        assert_eq!(config.padding_color, custom_padding);
        assert_eq!(config.normalization.mean, custom_norm.mean);
        assert_eq!(config.normalization.std, custom_norm.std);
    }
}
