use clashvision::session::yolo_session::YoloSession;
use criterion::{Criterion, criterion_group, criterion_main};

#[allow(dead_code)]
fn bench_process_image() {
    const IMAGE_PATH: &str = "assets/village_1759583099.png";
    const MODEL_PATH: &str = "models/best.onnx";

    let mut yolo_model: YoloSession =
        YoloSession::new(MODEL_PATH, "yolov8".into()).expect("Failed to create YOLO model");

    yolo_model
        .process_image(IMAGE_PATH)
        .expect("Failed to process image");
}

#[allow(dead_code)]
fn benchmark_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_application");
    group.bench_function("test_process_image", |b| {
        b.iter(|| {
            bench_process_image();
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_application);
criterion_main!(benches);
