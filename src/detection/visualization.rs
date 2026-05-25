//! Visualization utilities for drawing bounding boxes on images.

use super::bbox::BoundingBox;
use crate::class::clash_class::ClashClass;
use image::RgbImage;
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::sync::LazyLock;

/// Static palette indexed by `class_id`. Built once at first use.
static CLASS_COLORS: LazyLock<Vec<SolidSource>> = LazyLock::new(|| {
    ClashClass::rgb_colors()
        .iter()
        .map(|&(r, g, b, a)| SolidSource { r, g, b, a })
        .collect()
});

const FALLBACK_COLOR: SolidSource = SolidSource {
    r: 0x80,
    g: 0x10,
    b: 0x40,
    a: 0xFF,
};

#[inline]
fn color_for(class_id: usize) -> SolidSource {
    CLASS_COLORS.get(class_id).copied().unwrap_or(FALLBACK_COLOR)
}

/// Configuration for drawing bounding boxes.
#[derive(Debug, Clone, PartialEq)]
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

impl DrawConfig {
    /// Draws bounding boxes onto the given image (consumed, returned with boxes drawn).
    /// Zero-copy in the no-boxes case.
    #[must_use]
    pub fn draw_bounding_boxes(
        image: RgbImage,
        boxes: &[BoundingBox],
        input_size: (u32, u32),
        config: Option<DrawConfig>,
    ) -> RgbImage {
        if boxes.is_empty() {
            return image;
        }

        let config = config.unwrap_or_default();
        let (img_width, img_height) = image.dimensions();
        let mut draw_target = DrawTarget::new(img_width as i32, img_height as i32);

        let scale_x = img_width as f32 / input_size.0 as f32;
        let scale_y = img_height as f32 / input_size.1 as f32;

        for bbox in boxes {
            Self::draw_single_box(&mut draw_target, bbox, scale_x, scale_y, &config);
        }

        blend_into(image, draw_target, config.alpha_blend)
    }

    /// Draws a single bounding box on the draw target.
    fn draw_single_box(
        draw_target: &mut DrawTarget,
        bbox: &BoundingBox,
        scale_x: f32,
        scale_y: f32,
        config: &DrawConfig,
    ) {
        let mut path_builder = PathBuilder::new();

        let x = bbox.x1 * scale_x;
        let y = bbox.y1 * scale_y;
        let width = (bbox.x2 - bbox.x1) * scale_x;
        let height = (bbox.y2 - bbox.y1) * scale_y;

        path_builder.rect(x, y, width, height);
        let path = path_builder.finish();

        let color = color_for(bbox.class_id);

        let stroke_style = StrokeStyle {
            join: LineJoin::Round,
            width: config.line_width,
            ..StrokeStyle::default()
        };

        draw_target.stroke(
            &path,
            &Source::Solid(color),
            &stroke_style,
            &DrawOptions::new(),
        );
    }

    /// Convenience wrapper using default DrawConfig.
    #[must_use]
    pub fn draw_boxes(
        image: RgbImage,
        boxes: &[BoundingBox],
        input_size: (u32, u32),
    ) -> RgbImage {
        Self::draw_bounding_boxes(image, boxes, input_size, None)
    }
}

/// Blends the BGRA draw target into the RGB image buffer in place.
fn blend_into(mut image: RgbImage, draw_target: DrawTarget, alpha_blend: bool) -> RgbImage {
    if !alpha_blend {
        return image;
    }

    let bgra_data = draw_target.into_vec();
    let buf = image.as_mut();

    for (i, &pixel) in bgra_data.iter().enumerate() {
        let a = (pixel >> 24) & 0xFF;
        if a == 0 {
            continue;
        }

        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;
        let inv_a = 255 - a;

        let dst = i * 3;
        buf[dst] = ((r * a + buf[dst] as u32 * inv_a) / 255) as u8;
        buf[dst + 1] = ((g * a + buf[dst + 1] as u32 * inv_a) / 255) as u8;
        buf[dst + 2] = ((b * a + buf[dst + 2] as u32 * inv_a) / 255) as u8;
    }

    image
}
