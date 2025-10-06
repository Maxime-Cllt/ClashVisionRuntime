//! Visualization utilities for drawing bounding boxes on images.

use super::bbox::BoundingBox;
use crate::image::image_util::generate_class_colors;
use image::{DynamicImage, RgbImage, RgbaImage};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::collections::HashMap;

/// Configuration for drawing bounding boxes.
#[derive(Debug, Clone)]
pub struct DrawConfig {
    pub line_width: f32,
    pub alpha_blend: bool,
    pub show_confidence: bool,
    pub font_size: f32,
}

impl Default for DrawConfig {
    fn default() -> Self {
        Self {
            line_width: 4.0,
            alpha_blend: true,
            show_confidence: false,
            font_size: 12.0,
        }
    }
}

/// Draws bounding boxes on an image with improved performance and customization.
#[must_use]
pub fn draw_bounding_boxes(
    image: &DynamicImage,
    boxes: &[BoundingBox],
    input_size: (u32, u32),
    config: Option<DrawConfig>,
) -> RgbImage {
    let config = config.unwrap_or_default();
    let (img_width, img_height) = (image.width(), image.height());

    if boxes.is_empty() {
        return image.to_rgb8();
    }

    let mut draw_target = DrawTarget::new(img_width as i32, img_height as i32);
    let class_colors: HashMap<usize, SolidSource> = generate_colors_for_boxes(boxes);

    // Pre-calculate scaling factors
    let scale_x = img_width as f32 / input_size.0 as f32;
    let scale_y = img_height as f32 / input_size.1 as f32;

    for bbox in boxes {
        draw_single_box(
            &mut draw_target,
            bbox,
            &class_colors,
            scale_x,
            scale_y,
            &config,
        );
    }

    blend_with_original_image(image, draw_target, config.alpha_blend)
}

/// Generates colors for all unique classes in the bounding boxes.
fn generate_colors_for_boxes(boxes: &[BoundingBox]) -> HashMap<usize, SolidSource> {
    if boxes.is_empty() {
        return HashMap::new();
    }

    let unique_classes: std::collections::HashSet<usize> =
        boxes.iter().map(|bbox| bbox.class_id).collect();

    if unique_classes.is_empty() {
        return HashMap::new();
    }

    generate_class_colors()
}

/// Draws a single bounding box on the draw target.
fn draw_single_box(
    draw_target: &mut DrawTarget,
    bbox: &BoundingBox,
    class_colors: &HashMap<usize, SolidSource>,
    scale_x: f32,
    scale_y: f32,
    config: &DrawConfig,
) {
    let mut path_builder = PathBuilder::new();

    // Calculate scaled coordinates
    let x = bbox.x1 * scale_x;
    let y = bbox.y1 * scale_y;
    let width = (bbox.x2 - bbox.x1) * scale_x;
    let height = (bbox.y2 - bbox.y1) * scale_y;

    path_builder.rect(x, y, width, height);
    let path = path_builder.finish();

    // Get color for this class, with fallback
    let color = class_colors.get(&bbox.class_id).unwrap_or(&SolidSource {
        r: 0x80,
        g: 0x10,
        b: 0x40,
        a: 0xFF,
    });

    #[cfg(debug_assertions)]{
        println!(
            "Drawing box: class_id={}, x1={}, y1={}, x2={}, y2={}, color={:?}",
            bbox.class_id, bbox.x1, bbox.y1, bbox.x2, bbox.y2, color
        );
    }

    let stroke_style = StrokeStyle {
        join: LineJoin::Round,
        width: config.line_width,
        ..StrokeStyle::default()
    };

    draw_target.stroke(
        &path,
        &Source::Solid(*color),
        &stroke_style,
        &DrawOptions::new(),
    );
}

/// Blends the drawn boxes with the original image.
fn blend_with_original_image(
    original: &DynamicImage,
    draw_target: DrawTarget,
    alpha_blend: bool,
) -> RgbImage {
    let (img_width, img_height) = (original.width(), original.height());

    let box_image_rgba = RgbaImage::from_raw(
        img_width,
        img_height,
        draw_target
            .into_vec()
            .into_iter()
            .flat_map(|pixel| u32::to_ne_bytes(pixel))
            .collect(),
    )
    .expect("Failed to create RGBA image from draw target");

    let mut result = original.to_rgb8();

    if !alpha_blend {
        return result;
    }

    // Optimized alpha blending
    for (x, y, rgba_pixel) in box_image_rgba.enumerate_pixels() {
        let alpha = u16::from(rgba_pixel[3]);

        if alpha == 0 {
            continue; // Skip transparent pixels
        }

        let original_pixel = result.get_pixel_mut(x, y);
        let inv_alpha = 255 - alpha;

        // Blend each color channel
        for i in 0..3 {
            original_pixel[i] = u8::try_from(
                (u16::from(rgba_pixel[i]) * alpha + u16::from(original_pixel[i]) * inv_alpha) / 255,
            )
            .unwrap_or(0);
        }
    }

    result
}

// Backward compatibility function
#[must_use]
pub fn draw_boxes(image: &DynamicImage, boxes: &[BoundingBox], input_size: (u32, u32)) -> RgbImage {
    draw_bounding_boxes(image, boxes, input_size, None)
}
