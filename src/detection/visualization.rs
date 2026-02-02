//! Visualization utilities for drawing bounding boxes on images.

use super::bbox::BoundingBox;
use crate::image::image_util::generate_class_colors;
use image::{DynamicImage, RgbImage};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::collections::HashMap;

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
        let class_colors: HashMap<usize, SolidSource> = Self::generate_colors_for_boxes(boxes);

        // Pre-calculate scaling factors
        let scale_x = img_width as f32 / input_size.0 as f32;
        let scale_y = img_height as f32 / input_size.1 as f32;

        for bbox in boxes {
            Self::draw_single_box(
                &mut draw_target,
                bbox,
                &class_colors,
                scale_x,
                scale_y,
                &config,
            );
        }

        Self::blend_with_original_image(image, draw_target, config.alpha_blend)
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

        #[cfg(debug_assertions)]
        {
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

        // Draw the rectangle on the draw target
        draw_target.stroke(
            &path,
            &Source::Solid(*color),
            &stroke_style,
            &DrawOptions::new(),
        );
    }

    // Backward compatibility function
    #[must_use]
    pub fn draw_boxes(
        image: &DynamicImage,
        boxes: &[BoundingBox],
        input_size: (u32, u32),
    ) -> RgbImage {
        Self::draw_bounding_boxes(image, boxes, input_size, None)
    }

    /// Generates colors for all unique classes in the bounding boxes.
    fn generate_colors_for_boxes(boxes: &[BoundingBox]) -> HashMap<usize, SolidSource> {
        if boxes.is_empty() {
            return HashMap::new();
        }

        // Get all class colors from ClashClass
        let all_class_colors = generate_class_colors();

        // Only return colors for classes that are actually present in the boxes
        let unique_classes: std::collections::HashSet<usize> =
            boxes.iter().map(|bbox| bbox.class_id).collect();

        if unique_classes.is_empty() {
            return HashMap::new();
        }

        // Filter to only include colors for classes present in the boxes
        unique_classes
            .into_iter()
            .filter_map(|class_id| {
                all_class_colors
                    .get(&class_id)
                    .map(|&color| (class_id, color))
            })
            .collect()
    }

    /// Blends the drawn boxes with the original image.
    fn blend_with_original_image(
        original: &DynamicImage,
        draw_target: DrawTarget,
        alpha_blend: bool,
    ) -> RgbImage {
        let mut result = original.to_rgb8();

        if !alpha_blend {
            return result;
        }

        // Process raw BGRA u32 buffer directly, blending into the RGB result
        let bgra_data = draw_target.into_vec();
        let result_buf = result.as_mut();

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
            result_buf[dst] = ((r * a + result_buf[dst] as u32 * inv_a) / 255) as u8;
            result_buf[dst + 1] = ((g * a + result_buf[dst + 1] as u32 * inv_a) / 255) as u8;
            result_buf[dst + 2] = ((b * a + result_buf[dst + 2] as u32 * inv_a) / 255) as u8;
        }

        result
    }
}
