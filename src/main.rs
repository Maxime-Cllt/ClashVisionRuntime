use clashvision::MODEL_BYTES;
use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

fn main() {
    let args: Vec<String> = std::env::args().collect::<Vec<String>>();
    if args.len() < 2 {
        eprintln!("Usage cargo run --: {} <image_path>", args[0]);
        panic!("Not enough arguments");
    }

    let image_path: String = args[1].clone();

    // Use the embedded model bytes
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, YoloType::YoloV8)
        .expect("Failed to create YOLO model from embedded bytes");

    yolo_model
        .process_image(&image_path)
        .expect("Failed to process image");
}
