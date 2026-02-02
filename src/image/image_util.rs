use crate::class::clash_class::ClashClass;
use crate::image::image_config::ImageConfig;
use crate::image::image_size::ImageSize;
use crate::image::loaded_image::{LoadedImageF32, LoadedImageU8};
use crate::image::{DEFAULT_MEAN, DEFAULT_STD};
use image::{ImageBuffer, ImageError, Rgb};
use ndarray::Array4;
use raqote::SolidSource;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ImageLoadError {
    #[error("Image processing error: {0}")]
    ImageError(#[from] ImageError),
    #[error("Invalid image path: {0}")]
    InvalidPath(String),
}

/// Loads and preprocesses an image from the specified path
pub fn load_image_u8(
    image_path: impl AsRef<Path>,
    config: &ImageConfig,
) -> Result<LoadedImageU8, ImageLoadError> {
    let image_path = image_path.as_ref();

    if !image_path.exists() {
        return Err(ImageLoadError::InvalidPath(
            image_path.display().to_string(),
        ));
    }

    let image = image::open(image_path)?;
    let resized_padded = resize_and_pad_image(&image, config);
    let array = image_to_array(&resized_padded, config.target_size);

    Ok(LoadedImageU8::new(array, config.target_size))
}

/// Convenience function with default configuration
pub fn load_image_u8_default(
    image_path: impl AsRef<Path>,
    target_size: (u32, u32),
) -> Result<LoadedImageU8, ImageLoadError> {
    let config = ImageConfig {
        target_size: ImageSize::new(target_size.0, target_size.1),
        ..Default::default()
    };
    load_image_u8(image_path, &config)
}

/// Resizes image while maintaining aspect ratio and adds padding
fn resize_and_pad_image(
    image: &image::DynamicImage,
    config: &ImageConfig,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (orig_width, orig_height) = (image.width(), image.height());
    let target_size = config.target_size;

    // Calculate scale to maintain aspect ratio
    let scale_x = target_size.width as f32 / orig_width as f32;
    let scale_y = target_size.height as f32 / orig_height as f32;
    let scale = scale_x.min(scale_y);

    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    // Resize image
    let resized_image = image
        .resize_exact(new_width, new_height, config.filter_type)
        .to_rgb8();

    // Calculate padding
    let pad_left = (target_size.width - new_width) / 2;
    let pad_top = (target_size.height - new_height) / 2;

    // Create padded image
    let padding_pixel = Rgb(config.padding_color);
    let mut padded_image =
        ImageBuffer::from_pixel(target_size.width, target_size.height, padding_pixel);

    // Copy resized image to center using direct row-based memcpy
    let row_bytes = (new_width as usize) * 3;
    let target_stride = (target_size.width as usize) * 3;
    let src_buf = resized_image.as_raw();
    let dst_buf = padded_image.as_mut();

    for y in 0..new_height as usize {
        let src_offset = y * row_bytes;
        let dst_offset = (y + pad_top as usize) * target_stride + (pad_left as usize) * 3;
        dst_buf[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&src_buf[src_offset..src_offset + row_bytes]);
    }

    padded_image
}

/// Converts `ImageBuffer` to ndarray with NCHW format
fn image_to_array(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, size: ImageSize) -> Array4<u8> {
    let h = size.height as usize;
    let w = size.width as usize;
    let hw = h * w;
    let raw = image.as_raw();

    // Pre-allocate flat buffer for NCHW layout and fill in a single pass
    let mut data = vec![0u8; 3 * hw];
    let (ch_r, rest) = data.split_at_mut(hw);
    let (ch_g, ch_b) = rest.split_at_mut(hw);

    for i in 0..hw {
        let src = i * 3;
        ch_r[i] = raw[src];
        ch_g[i] = raw[src + 1];
        ch_b[i] = raw[src + 2];
    }

    Array4::from_shape_vec((1, 3, h, w), data).expect("Failed to create NCHW array")
}

/// Normalizes the image using the provided mean and std deviation.
pub fn normalize_image_f32(
    loaded_image: &LoadedImageU8,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) -> LoadedImageF32 {
    let mean = mean.unwrap_or(DEFAULT_MEAN);
    let std = std.unwrap_or(DEFAULT_STD);

    let shape = loaded_image.image_array.shape();
    let h = shape[2];
    let w = shape[3];
    let hw = h * w;

    // Pre-compute scale and offset per channel: result = (x / 255.0 - mean) / std = x * scale + offset
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c]));
    let offset: [f32; 3] = std::array::from_fn(|c| -mean[c] / std[c]);

    let src = loaded_image.image_array.as_slice().unwrap();
    let mut data = vec![0.0f32; 3 * hw];

    for c in 0..3 {
        let s = scale[c];
        let o = offset[c];
        let src_slice = &src[c * hw..(c + 1) * hw];
        let dst_slice = &mut data[c * hw..(c + 1) * hw];
        for i in 0..hw {
            dst_slice[i] = src_slice[i] as f32 * s + o;
        }
    }

    let array = Array4::from_shape_vec((1, 3, h, w), data)
        .expect("Failed to create normalized array");

    LoadedImageF32 {
        image_array: array,
        size: loaded_image.size,
    }
}

/// Generates distinct colors for each class using a more sophisticated color scheme
#[must_use]
pub fn generate_class_colors() -> HashMap<usize, SolidSource> {
    let num_classes = ClashClass::num_classes();
    let mut class_colors = HashMap::with_capacity(num_classes);

    // Use predefined colors if available
    let predefined_colors = ClashClass::rgb_colors();

    for (i, &color) in predefined_colors.iter().enumerate() {
        let (r, g, b, a) = color;
        class_colors.insert(i, SolidSource { r, g, b, a });
    }

    class_colors
}

/// Generates colors using HSV color space for better distribution
#[must_use]
pub fn generate_distinct_colors(num_colors: usize) -> Vec<SolidSource> {
    (0..num_colors)
        .map(|i| {
            let hue = (i as f32 * 360.0 / num_colors as f32) % 360.0;
            let saturation = 0.7;
            let value = 0.9;

            let (r, g, b) = hsv_to_rgb(hue, saturation, value);

            SolidSource {
                r: (r * 255.0) as u8,
                g: (g * 255.0) as u8,
                b: (b * 255.0) as u8,
                a: 255,
            }
        })
        .collect()
}

/// Converts HSV color space to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r_prime, g_prime, b_prime) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0), // Should never happen with proper input
    };

    (r_prime + m, g_prime + m, b_prime + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsv_to_rgb() {
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0); // Pure red
        assert!((r - 1.0).abs() < f32::EPSILON);
        assert!(g.abs() < f32::EPSILON);
        assert!(b.abs() < f32::EPSILON);
    }
}
