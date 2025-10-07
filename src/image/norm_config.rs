use crate::image::{DEFAULT_MEAN, DEFAULT_STD, IMAGENET_MEAN, IMAGENET_STD};

#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl NormalizationConfig {
    /// Predefined normalization config for `ImageNet` models
    #[inline]
    #[must_use]
    pub const fn imagenet() -> Self {
        Self {
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }

    /// No normalization (mean=0, std=1)
    #[inline]
    #[must_use]
    pub const fn none() -> Self {
        Self {
            mean: DEFAULT_MEAN,
            std: DEFAULT_STD,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{DEFAULT_MEAN, DEFAULT_STD, IMAGENET_MEAN, IMAGENET_STD};

    #[test]
    fn test_imagenet_normalization() {
        let config = NormalizationConfig::imagenet();
        assert_eq!(config.mean, IMAGENET_MEAN);
        assert_eq!(config.std, IMAGENET_STD);
    }

    #[test]
    fn test_no_normalization() {
        let config = NormalizationConfig::none();
        assert_eq!(config.mean, DEFAULT_MEAN);
        assert_eq!(config.std, DEFAULT_STD);
    }
}
