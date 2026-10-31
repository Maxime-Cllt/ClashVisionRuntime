use clashvision::MODEL_BYTES;
use clashvision::image::image_util::load_and_preprocess_f32;
use clashvision::model::yolo_type::YoloType;
use clashvision::session::yolo_session::YoloSession;
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const IMAGE_PATH: &str = "assets/village_1759583099.png";
const INPUT_SIZE: (u32, u32) = (640, 640);

fn benchmark_application(c: &mut Criterion) {
    // Session creation (model load + EP init) happens ONCE, outside the timed loop.
    // The previous version recreated the session per iteration, so the benchmark was
    // dominated by model loading, not inference.
    let mut yolo_model = YoloSession::from_bytes(MODEL_BYTES, YoloType::YoloV8)
        .expect("Failed to create YOLO model from embedded bytes");

    let mut group = c.benchmark_group("benchmark_application");

    group.bench_function("preprocess", |b| {
        b.iter(|| {
            load_and_preprocess_f32(IMAGE_PATH, INPUT_SIZE, None, None)
                .expect("Failed to preprocess image")
        })
    });

    let (_, loaded) = load_and_preprocess_f32(IMAGE_PATH, INPUT_SIZE, None, None)
        .expect("Failed to preprocess image");
    group.bench_function("inference", |b| {
        b.iter(|| {
            yolo_model
                .run_inference(loaded.image_array.clone())
                .expect("Failed to run inference")
        })
    });

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
