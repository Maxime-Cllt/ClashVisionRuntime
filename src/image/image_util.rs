use crate::image::image_config::ImageConfig;
use crate::image::image_size::ImageSize;
use crate::image::loaded_image::LoadedImageF32;
use crate::image::{DEFAULT_MEAN, DEFAULT_STD};
use fast_image_resize::{self as fr, ResizeOptions};
use image::{DynamicImage, ImageBuffer, ImageError, Rgb};
use ndarray::Array4;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ImageLoadError {
    #[error("Image processing error: {0}")]
    ImageError(#[from] ImageError),
    #[error("Invalid image path: {0}")]
    InvalidPath(String),
    #[error("Resize error: {0}")]
    ResizeError(String),
}

type RgbBuffer = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Loads, resizes, pads, and normalizes an image in a single fused pipeline.
/// Returns the normalized f32 NCHW tensor directly (no intermediate u8 array).
/// Also returns the resized+padded RGB image for visualization.
pub fn load_and_preprocess_f32(
    image_path: impl AsRef<Path>,
    target_size: (u32, u32),
    mean: Option<[f32; 3]>,
    std_dev: Option<[f32; 3]>,
) -> Result<(RgbBuffer, LoadedImageF32), ImageLoadError> {
    let image_path = image_path.as_ref();

    if !image_path.exists() {
        return Err(ImageLoadError::InvalidPath(
            image_path.display().to_string(),
        ));
    }

    let image = image::open(image_path)?;
    let config = ImageConfig {
        target_size: ImageSize::new(target_size.0, target_size.1),
        ..Default::default()
    };

    let resized_padded = resize_and_pad_image_fast(&image, &config)?;
    let normalized = fused_normalize_nchw(&resized_padded, config.target_size, mean, std_dev);

    Ok((resized_padded, normalized))
}

/// SIMD-accelerated resize (fast_image_resize) + centered letterbox padding via memcpy.
fn resize_and_pad_image_fast(
    image: &DynamicImage,
    config: &ImageConfig,
) -> Result<RgbBuffer, ImageLoadError> {
    let (orig_width, orig_height) = (image.width(), image.height());
    let target_size = config.target_size;

    let scale_x = target_size.width as f32 / orig_width as f32;
    let scale_y = target_size.height as f32 / orig_height as f32;
    let scale = scale_x.min(scale_y);

    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    let src_rgb = image.to_rgb8();

    let src_image = fr::images::Image::from_vec_u8(
        orig_width,
        orig_height,
        src_rgb.into_raw(),
        fr::PixelType::U8x3,
    )
    .map_err(|e| ImageLoadError::ResizeError(format!("Failed to create source image: {e}")))?;

    let mut dst_image = fr::images::Image::new(new_width, new_height, fr::PixelType::U8x3);

    let mut resizer = fr::Resizer::new();
    let options =
        ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .map_err(|e| ImageLoadError::ResizeError(format!("Resize failed: {e}")))?;

    let resized_data = dst_image.into_vec();

    let padding_pixel = Rgb(config.padding_color);
    let mut padded_image =
        ImageBuffer::from_pixel(target_size.width, target_size.height, padding_pixel);

    let pad_left = (target_size.width - new_width) / 2;
    let pad_top = (target_size.height - new_height) / 2;

    let row_bytes = new_width as usize * 3;
    let target_stride = target_size.width as usize * 3;
    let dst_buf = padded_image.as_mut();

    for y in 0..new_height as usize {
        let src_offset = y * row_bytes;
        let dst_offset = (y + pad_top as usize) * target_stride + (pad_left as usize) * 3;
        dst_buf[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&resized_data[src_offset..src_offset + row_bytes]);
    }

    Ok(padded_image)
}

/// Fused HWC→NCHW conversion + f32 normalization in a single pass.
/// Uses `chunks_exact(3)` so the compiler can autovectorize the per-pixel work.
fn fused_normalize_nchw(
    image: &RgbBuffer,
    size: ImageSize,
    mean: Option<[f32; 3]>,
    std_dev: Option<[f32; 3]>,
) -> LoadedImageF32 {
    let mean = mean.unwrap_or(DEFAULT_MEAN);
    let std = std_dev.unwrap_or(DEFAULT_STD);

    let h = size.height as usize;
    let w = size.width as usize;
    let hw = h * w;
    let raw = image.as_raw();

    // result = (x / 255.0 - mean) / std = x * scale + offset
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c]));
    let offset: [f32; 3] = std::array::from_fn(|c| -mean[c] / std[c]);

    let mut data: Vec<f32> = vec![0.0; 3 * hw];
    let (ch_r, rest) = data.split_at_mut(hw);
    let (ch_g, ch_b) = rest.split_at_mut(hw);

    let (sr, sg, sb) = (scale[0], scale[1], scale[2]);
    let (or, og, ob) = (offset[0], offset[1], offset[2]);

    for (i, pixel) in raw.chunks_exact(3).enumerate() {
        ch_r[i] = f32::from(pixel[0]) * sr + or;
        ch_g[i] = f32::from(pixel[1]) * sg + og;
        ch_b[i] = f32::from(pixel[2]) * sb + ob;
    }

    let array = Array4::from_shape_vec((1, 3, h, w), data)
        .expect("Failed to create normalized NCHW array");

    LoadedImageF32 {
        image_array: array,
        size,
    }
}
