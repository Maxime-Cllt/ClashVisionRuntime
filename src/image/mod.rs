pub mod image_config;
mod image_size;
pub mod image_util;
pub mod loaded_image;
mod norm_config;

// ImageNet normalization constants - commonly used in computer vision
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// Fallback normalization constants
const DEFAULT_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_STD: [f32; 3] = [1.0, 1.0, 1.0];

// Padding color (gray)
const PADDING_COLOR: [u8; 3] = [112, 112, 112];
