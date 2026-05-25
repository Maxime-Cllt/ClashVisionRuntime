use clashvision::MODEL_BYTES;
use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;
use criterion::{Criterion, criterion_group, criterion_main};

const IMAGE_PATH: &str = "assets/village_1759583099.png";

fn benchmark_application(c: &mut Criterion) {
    // Session creation (model load + EP init) happens ONCE, outside the timed loop.
    // The previous version recreated the session per iteration, so the benchmark was
    // dominated by model loading, not inference.
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, YoloType::YoloV8)
        .expect("Failed to create YOLO model from embedded bytes");

    let mut group = c.benchmark_group("benchmark_application");
    group.bench_function("process_image", |b| {
        b.iter(|| {
            yolo_model
                .process_image(IMAGE_PATH)
                .expect("Failed to process image");
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_application);
criterion_main!(benches);
