use crate::image::image_size::ImageSize;
use crate::image::loaded_image::LoadedImageF32;
use crate::image::{DEFAULT_MEAN, DEFAULT_STD, PADDING_COLOR};
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

/// Channel layout of a raw 8-bit pixel buffer (alpha, if any, is ignored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb,
    Rgba,
}

impl PixelFormat {
    const fn channels(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    const fn pixel_type(self) -> fr::PixelType {
        match self {
            Self::Rgb => fr::PixelType::U8x3,
            Self::Rgba => fr::PixelType::U8x4,
        }
    }
}

/// Where the resized image sits inside the letterboxed square target.
struct Letterbox {
    target: ImageSize,
    new_width: u32,
    new_height: u32,
    pad_left: usize,
    pad_top: usize,
}

impl Letterbox {
    fn new(orig_width: u32, orig_height: u32, target: ImageSize) -> Self {
        let scale_x = target.width as f32 / orig_width as f32;
        let scale_y = target.height as f32 / orig_height as f32;
        let scale = scale_x.min(scale_y);

        let new_width = ((orig_width as f32 * scale).round() as u32).clamp(1, target.width);
        let new_height = ((orig_height as f32 * scale).round() as u32).clamp(1, target.height);

        Self {
            target,
            new_width,
            new_height,
            pad_left: ((target.width - new_width) / 2) as usize,
            pad_top: ((target.height - new_height) / 2) as usize,
        }
    }
}

/// Loads, resizes, pads, and normalizes an image file in a single fused pipeline.
/// Returns the normalized f32 NCHW tensor directly (no intermediate u8 array).
/// Also returns the resized+padded RGB image for visualization.
///
/// If you already have the pixels in memory (e.g. a screenshot), prefer
/// [`preprocess_image`] / [`preprocess_raw`]: they skip the encode/decode round-trip
/// through a file and don't build the visualization image.
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
    let target = ImageSize::new(target_size.0, target_size.1);

    let (resized, format, letterbox) = resize_dynamic(&image, target)?;
    let normalized = normalize_letterboxed_nchw(&resized, format, &letterbox, mean, std_dev);
    let padded = build_padded_image(&resized, format, &letterbox);

    Ok((padded, normalized))
}

/// Resizes (letterbox) and normalizes an in-memory image into an NCHW f32 tensor.
/// RGB8 / RGBA8 images are read in place (no conversion copy).
pub fn preprocess_image(
    image: &DynamicImage,
    target_size: (u32, u32),
    mean: Option<[f32; 3]>,
    std_dev: Option<[f32; 3]>,
) -> Result<LoadedImageF32, ImageLoadError> {
    let target = ImageSize::new(target_size.0, target_size.1);
    let (resized, format, letterbox) = resize_dynamic(image, target)?;
    Ok(normalize_letterboxed_nchw(
        &resized, format, &letterbox, mean, std_dev,
    ))
}

/// Same as [`preprocess_image`] for a raw tightly-packed 8-bit pixel buffer
/// (e.g. the bytes of an `image::RgbaImage`).
pub fn preprocess_raw(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    target_size: (u32, u32),
    mean: Option<[f32; 3]>,
    std_dev: Option<[f32; 3]>,
) -> Result<LoadedImageF32, ImageLoadError> {
    let target = ImageSize::new(target_size.0, target_size.1);
    let (resized, letterbox) = resize_raw(pixels, width, height, format, target)?;
    Ok(normalize_letterboxed_nchw(
        &resized, format, &letterbox, mean, std_dev,
    ))
}

/// Picks the cheapest zero-copy view of `image` and resizes it.
fn resize_dynamic(
    image: &DynamicImage,
    target: ImageSize,
) -> Result<(Vec<u8>, PixelFormat, Letterbox), ImageLoadError> {
    let (width, height) = (image.width(), image.height());
    match image {
        DynamicImage::ImageRgb8(img) => {
            let (resized, lb) = resize_raw(img.as_raw(), width, height, PixelFormat::Rgb, target)?;
            Ok((resized, PixelFormat::Rgb, lb))
        }
        DynamicImage::ImageRgba8(img) => {
            let (resized, lb) =
                resize_raw(img.as_raw(), width, height, PixelFormat::Rgba, target)?;
            Ok((resized, PixelFormat::Rgba, lb))
        }
        other => {
            let rgb = other.to_rgb8();
            let (resized, lb) = resize_raw(rgb.as_raw(), width, height, PixelFormat::Rgb, target)?;
            Ok((resized, PixelFormat::Rgb, lb))
        }
    }
}

/// SIMD-accelerated resize (fast_image_resize) of a borrowed pixel buffer.
fn resize_raw(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    target: ImageSize,
) -> Result<(Vec<u8>, Letterbox), ImageLoadError> {
    let letterbox = Letterbox::new(width, height, target);

    let src_image = fr::images::ImageRef::new(width, height, pixels, format.pixel_type())
        .map_err(|e| ImageLoadError::ResizeError(format!("Failed to create source image: {e}")))?;
    let mut dst_image = fr::images::Image::new(
        letterbox.new_width,
        letterbox.new_height,
        format.pixel_type(),
    );

    let mut resizer = fr::Resizer::new();
    let options =
        ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .map_err(|e| ImageLoadError::ResizeError(format!("Resize failed: {e}")))?;

    Ok((dst_image.into_vec(), letterbox))
}

/// Builds the letterboxed RGB image (gray padding) used for visualization.
fn build_padded_image(resized: &[u8], format: PixelFormat, lb: &Letterbox) -> RgbBuffer {
    let mut padded = ImageBuffer::from_pixel(lb.target.width, lb.target.height, Rgb(PADDING_COLOR));

    let channels = format.channels();
    let target_stride = lb.target.width as usize * 3;
    let dst_buf = padded.as_mut();

    for (y, src_row) in resized
        .chunks_exact(lb.new_width as usize * channels)
        .enumerate()
    {
        let dst_offset = (y + lb.pad_top) * target_stride + lb.pad_left * 3;
        let dst_row = &mut dst_buf[dst_offset..dst_offset + lb.new_width as usize * 3];
        if channels == 3 {
            dst_row.copy_from_slice(src_row);
        } else {
            for (dst, src) in dst_row.chunks_exact_mut(3).zip(src_row.chunks_exact(channels)) {
                dst.copy_from_slice(&src[..3]);
            }
        }
    }

    padded
}

/// Fused letterbox padding + HWC→NCHW conversion + f32 normalization in a single pass,
/// writing straight into the output tensor (no intermediate padded image).
fn normalize_letterboxed_nchw(
    resized: &[u8],
    format: PixelFormat,
    lb: &Letterbox,
    mean: Option<[f32; 3]>,
    std_dev: Option<[f32; 3]>,
) -> LoadedImageF32 {
    let mean = mean.unwrap_or(DEFAULT_MEAN);
    let std = std_dev.unwrap_or(DEFAULT_STD);

    let h = lb.target.height as usize;
    let w = lb.target.width as usize;
    let hw = h * w;
    let channels = format.channels();

    // result = (x / 255.0 - mean) / std = x * scale + offset
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c]));
    let offset: [f32; 3] = std::array::from_fn(|c| -mean[c] / std[c]);

    let mut data: Vec<f32> = vec![0.0; 3 * hw];
    let (ch_r, rest) = data.split_at_mut(hw);
    let (ch_g, ch_b) = rest.split_at_mut(hw);

    // Padding value of every channel, normalized exactly like a regular pixel.
    ch_r.fill(f32::from(PADDING_COLOR[0]) * scale[0] + offset[0]);
    ch_g.fill(f32::from(PADDING_COLOR[1]) * scale[1] + offset[1]);
    ch_b.fill(f32::from(PADDING_COLOR[2]) * scale[2] + offset[2]);

    let new_width = lb.new_width as usize;
    let (sr, sg, sb) = (scale[0], scale[1], scale[2]);
    let (or, og, ob) = (offset[0], offset[1], offset[2]);

    for (y, src_row) in resized.chunks_exact(new_width * channels).enumerate() {
        let start = (y + lb.pad_top) * w + lb.pad_left;
        let end = start + new_width;
        let (row_r, row_g, row_b) = (
            &mut ch_r[start..end],
            &mut ch_g[start..end],
            &mut ch_b[start..end],
        );

        // `chunks_exact` with a compile-time-constant size lets the compiler autovectorize.
        match format {
            PixelFormat::Rgb => {
                for (i, px) in src_row.chunks_exact(3).enumerate() {
                    row_r[i] = f32::from(px[0]) * sr + or;
                    row_g[i] = f32::from(px[1]) * sg + og;
                    row_b[i] = f32::from(px[2]) * sb + ob;
                }
            }
            PixelFormat::Rgba => {
                for (i, px) in src_row.chunks_exact(4).enumerate() {
                    row_r[i] = f32::from(px[0]) * sr + or;
                    row_g[i] = f32::from(px[1]) * sg + og;
                    row_b[i] = f32::from(px[2]) * sb + ob;
                }
            }
        }
    }

    let array = Array4::from_shape_vec((1, 3, h, w), data)
        .expect("Failed to create normalized NCHW array");

    LoadedImageF32 {
        image_array: array,
        size: lb.target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn gradient_rgba(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, 77, 255]))
    }

    #[test]
    fn rgba_and_rgb_inputs_give_identical_tensors() {
        let rgba = gradient_rgba(200, 100);
        let rgb = DynamicImage::ImageRgba8(rgba.clone()).to_rgb8();

        let from_rgba = preprocess_image(&DynamicImage::ImageRgba8(rgba), (64, 64), None, None).unwrap();
        let from_rgb = preprocess_image(&DynamicImage::ImageRgb8(rgb), (64, 64), None, None).unwrap();

        assert_eq!(from_rgba.shape(), &[1, 3, 64, 64]);
        assert_eq!(from_rgba.image_array, from_rgb.image_array);
    }

    #[test]
    fn letterbox_padding_is_normalized_gray() {
        // 200x100 into 64x64 → 64x32 image, 16 rows of padding above and below.
        let tensor = preprocess_image(
            &DynamicImage::ImageRgba8(gradient_rgba(200, 100)),
            (64, 64),
            None,
            None,
        )
        .unwrap()
        .image_array;

        let gray = f32::from(PADDING_COLOR[0]) / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - gray).abs() < 1e-6);
        assert!((tensor[[0, 0, 63, 63]] - gray).abs() < 1e-6);
        assert!((tensor[[0, 0, 32, 32]] - gray).abs() > 1e-3);
    }

    #[test]
    fn tensor_matches_visualization_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("img.png");
        gradient_rgba(120, 90).save(&path).unwrap();

        let (padded, loaded) = load_and_preprocess_f32(&path, (64, 64), None, None).unwrap();
        for (x, y) in [(0u32, 0u32), (10, 20), (63, 63), (32, 32)] {
            for c in 0..3 {
                let expected = f32::from(padded.get_pixel(x, y)[c]) / 255.0;
                let got = loaded.image_array[[0, c, y as usize, x as usize]];
                assert!((got - expected).abs() < 1e-6, "mismatch at ({x},{y}) c={c}");
            }
        }
    }

    #[test]
    fn preprocess_raw_matches_preprocess_image() {
        let rgba = gradient_rgba(50, 80);
        let raw = preprocess_raw(rgba.as_raw(), 50, 80, PixelFormat::Rgba, (32, 32), None, None).unwrap();
        let img = preprocess_image(&DynamicImage::ImageRgba8(rgba), (32, 32), None, None).unwrap();
        assert_eq!(raw.image_array, img.image_array);
    }
}
