use anyhow::{anyhow, Result};
use image::Rgb;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};
use clashvision::structs::detection::Detection;
use clashvision::structs::yolo_model::YOLOModel;

// Class names for your 2 classes - modify these according to your model
const CLASS_NAMES: [&str; 2] = ["class_0", "class_1"];

// Colors for different classes
const COLORS: [(u8, u8, u8); 2] = [
    (255, 0, 0), // Red for class 0
    (0, 255, 0), // Green for class 1
];

// Alternative version without external font dependency
fn draw_detections_simple(image_path: &str, detections: &[Detection], output_path: &str) -> Result<()> {
    let mut img = image::open(image_path)?.to_rgb8();

    for detection in detections {
        let bbox = &detection.bbox;
        let class_name = CLASS_NAMES[detection.class_id];
        let color = COLORS[detection.class_id];
        let rgb_color = Rgb([color.0, color.1, color.2]);

        // Draw bounding box (thicker lines)
        let rect = Rect::at(bbox[0] as i32, bbox[1] as i32)
            .of_size((bbox[2] - bbox[0]) as u32, (bbox[3] - bbox[1]) as u32);

        // Draw multiple rectangles for thicker lines
        for offset in 0..3 {
            let thick_rect = Rect::at(
                (bbox[0] as i32).saturating_sub(offset),
                (bbox[1] as i32).saturating_sub(offset)
            ).of_size(
                (bbox[2] - bbox[0]) as u32 + (2 * offset) as u32,
                (bbox[3] - bbox[1]) as u32 + (2 * offset) as u32
            );
            draw_hollow_rect_mut(&mut img, thick_rect, rgb_color);
        }

        println!(
            "Detected {}: confidence={:.3}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
            class_name, detection.confidence, bbox[0], bbox[1], bbox[2], bbox[3]
        );
    }

    img.save(output_path)?;
    println!("Saved result to: {}", output_path);

    Ok(())
}

fn main() -> Result<()> {
    let model_path = "/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/models/v1/best.torchscript";
    let input_image = "/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/data/images/val/village_1759335821.png"; // Change this to your input image path
    let output_image = "output_with_detections.jpg";

    println!("Loading YOLOv8 model...");
    let model = YOLOModel::new(model_path)?;

    println!("Running inference on image: {}", input_image);
    let detections = model.inference(input_image)?;

    println!("Found {} detections", detections.len());

    // Draw and save results
    draw_detections_simple(input_image, &detections, output_image)?;

    Ok(())
}
