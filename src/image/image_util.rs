use crate::class::clash_class::ClashClass;
use crate::image::image_config::ImageConfig;
use crate::image::image_size::ImageSize;
use crate::image::loaded_image::{LoadedImageF32, LoadedImageU8};
use crate::image::norm_config::NormalizationConfig;
use crate::image::{DEFAULT_MEAN, DEFAULT_STD};
use image::{ImageBuffer, ImageError, Pixel, Rgb};
use ndarray::{Array4, s};
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
    let resized_padded = resize_and_pad_image(image, config)?;
    let array = image_to_array(resized_padded, config.target_size);

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
    image: image::DynamicImage,
    config: &ImageConfig,
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, ImageLoadError> {
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

    // Copy resized image to center of padded image
    for (x, y, pixel) in resized_image.enumerate_pixels() {
        padded_image.put_pixel(x + pad_left, y + pad_top, *pixel);
    }

    Ok(padded_image)
}

/// Converts ImageBuffer to ndarray with NCHW format
fn image_to_array(image: ImageBuffer<Rgb<u8>, Vec<u8>>, size: ImageSize) -> Array4<u8> {
    Array4::from_shape_fn(
        (1, 3, size.height as usize, size.width as usize),
        |(_, c, y, x)| {
            let pixel = image.get_pixel(x as u32, y as u32);
            pixel.channels()[c]
        },
    )
}

/// Normalizes image from u8 to f32 with specified mean and standard deviation

/// Normalizes the image using the provided mean and std deviation.
pub fn normalize_image_f32(
    loaded_image: &LoadedImageU8,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) -> LoadedImageF32 {
    let mean = mean.unwrap_or(DEFAULT_MEAN);
    let std = std.unwrap_or(DEFAULT_STD);

    let mut array = loaded_image.image_array.mapv(|x| x as f32 / 255.0);

    for c in 0..3 {
        array
            .slice_mut(s![0, c, .., ..])
            .mapv_inplace(|x| (x - mean[c]) / std[c]);
    }

    LoadedImageF32 {
        image_array: array,
        size: loaded_image.size.clone(),
    }
}

/// Generates distinct colors for each class using a more sophisticated color scheme
pub fn generate_class_colors() -> HashMap<usize, SolidSource> {
    let num_classes = ClashClass::num_classes();
    let mut class_colors = HashMap::with_capacity(num_classes);

    // Use predefined colors if available
    let predefined_colors = ClashClass::colors();

    for (i, &color) in predefined_colors.iter().enumerate() {
        let (r, g, b, a) = color;
        class_colors.insert(i, SolidSource { r, g, b, a });
    }

    class_colors
}

/// Generates colors using HSV color space for better distribution
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
