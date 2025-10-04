use anyhow::Result;
use clashvision::enums::clash_class::ClashClass;
use clashvision::structs::detection::Detection;
use clashvision::structs::yolo_model::YOLOModel;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

// Class names for your 2 classes - modify these according to your model

// Colors for different classes
const COLORS: [(u8, u8, u8); 2] = [
    (255, 0, 0), // Red for class 0
    (0, 255, 0), // Green for class 1
];

// Alternative version without external font dependency
/// Draw bounding boxes safely
pub fn draw_detections_simple(image_path: &str, detections: &[Detection], output_path: &str) -> Result<()> {
    let mut img: RgbImage = image::open(image_path)?.to_rgb8();

    let class_names: Vec<&str> = ClashClass::values().iter().map(|c| c.as_str()).collect();

    for det in detections {
        let bbox = det.bbox;
        let class_id = det.class_id;


        let class_name = if class_id >= class_names.len() {
            "Unknown"
        }
        else { class_names[class_id] };


        let color = if class_id < COLORS.len() {
            COLORS[class_id]
        } else {
            (255, 255, 255) // White for unknown classes
        };
        let rgb_color = Rgb([color.0, color.1, color.2]);

        let rect = Rect::at(bbox[0] as i32, bbox[1] as i32)
            .of_size((bbox[2] - bbox[0]).max(1.0) as u32, (bbox[3] - bbox[1]).max(1.0) as u32);

        draw_hollow_rect_mut(&mut img, rect, rgb_color);

        println!(
            "Detected {}: conf={:.2}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
            class_name, det.confidence, bbox[0], bbox[1], bbox[2], bbox[3]
        );
    }

    img.save(output_path)?;
    println!("✅ Saved result to {}", output_path);
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
