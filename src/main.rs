use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

// Embed the model at compile time
const MODEL_BYTES: &[u8] = include_bytes!("../models/best.onnx");

fn main() {
    let args: Vec<String> = std::env::args().collect::<Vec<String>>();
    if args.len() < 2 {
        eprintln!("Usage cargo run --: {} <image_path>", args[0]);
        panic!("Not enough arguments");
    }

    let image_path: String = args[1].clone();
    let yolo_type: YoloType = YoloType::try_from("yolov8").expect("Failed to parse YOLO type");

    // Use the embedded model bytes instead of a file path
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, yolo_type)
        .expect("Failed to create YOLO model from embedded bytes");

    yolo_model
        .process_image(&image_path)
        .expect("Failed to process image");
}