use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

fn main() {
    const IMAGE_PATH: &str = "assets/village_1759583099.png";
    const MODEL_PATH: &str = "models/best.onnx";

    let start = std::time::Instant::now();
    let yolo_type: YoloType = YoloType::try_from("yolov8").expect("Failed to parse YOLO type");

    let mut yolo_model =
        YoloSession::new(MODEL_PATH, yolo_type).expect("Failed to create YOLO model");

    yolo_model
        .process_image(IMAGE_PATH)
        .expect("Failed to process image");

    let duration = start.elapsed();

    println!("Time elapsed in expensive_function() is: {:?}", duration);
}
