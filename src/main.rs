use clashvision::structs::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

fn main() {
    const IMAGE_PATH: &str = "assets/village_1759583099.png";
    const MODEL_PATH: &str = "models/best.onnx";

    let mut yolo_model = YoloSession::new(MODEL_PATH, (640, 640), false, "yolov8".to_string())
        .expect("Failed to create YOLO model");

    yolo_model.process_image(IMAGE_PATH);
}
