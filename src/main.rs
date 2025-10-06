use clashvision::structs::yolo_session::YoloSession;

#[cfg(test)]
mod benches;

fn main() {
    let image_path = "/Users/maximecolliat/Downloads/img/village_1759583271.png";

    let model_path: &str =
        "/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/models/v1/best.onnx";

    let mut yolo_model = YoloSession::new(model_path, (640, 640), false, "yolov8".to_string())
        .expect("Failed to create YOLO model");

    yolo_model.process_image(image_path);
}
