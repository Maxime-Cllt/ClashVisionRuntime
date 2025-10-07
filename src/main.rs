use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

fn main() {
    const MODEL_PATH: &str = "best.onnx";

    let args: Vec<String> = std::env::args().collect::<Vec<String>>();
    if args.len() < 2 {
        eprintln!("Usage cargo run --: {} <image_path>", args[0]);
        panic!("Not enough arguments");
    }

    let image_path: String = args[1].clone();
    let yolo_type: YoloType = YoloType::try_from("yolov8").expect("Failed to parse YOLO type");
    let mut yolo_model =
        YoloSession::new(MODEL_PATH, yolo_type).expect("Failed to create YOLO model");

    yolo_model
        .process_image(&image_path)
        .expect("Failed to process image");
}
