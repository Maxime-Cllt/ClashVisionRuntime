use clashvision::structs::yolo_session::YoloSession;
use criterion::{criterion_group, criterion_main, Criterion};

#[allow(dead_code)]
fn bench_process_image() {
    const IMAGE_PATH: &str = "assets/village_1759583099.png";
    const MODEL_PATH: &str = "models/v1/best.onnx";
    let mut yolo_model = YoloSession::new(MODEL_PATH, (640, 640), false, "yolov8".to_string())
        .expect("Failed to create YOLO model");

    yolo_model.process_image(IMAGE_PATH);
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
