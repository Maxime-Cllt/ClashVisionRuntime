use clashvision::MODEL_BYTES;
use clashvision::image::image_util::load_and_preprocess_f32;
use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;

#[test]
fn reference_village_has_detections() {
    let mut session = YoloSession::from_bytes(MODEL_BYTES, YoloType::YoloV8).expect("model should load");
    let (_, loaded) =
        load_and_preprocess_f32("assets/village_1759583099.png", (640, 640), None, None).unwrap();
    let boxes = session.run_inference(loaded.image_array).unwrap();
    assert!(!boxes.is_empty(), "the reference village should contain detections");
}
