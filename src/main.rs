use clashvision::structs::yolo_session::YoloSession;
use std::path::Path;

fn main() {
    let image_path = "/Users/maximecolliat/Downloads/img/village_1759583271.png";

    let model_path:&Path = Path::new("/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/models/v1/best.onnx");

    let model_name = "yolov8".to_string();

    let use_nms = false;

    let mut yolo_model = YoloSession::new(model_path, (640, 640), use_nms, model_name)
        .expect("Failed to create YOLO model");

    yolo_model.process_image(image_path);
}