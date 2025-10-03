use clashvision::enums::clash_class::ClashClass;
use clashvision::structs::yolo_model::YoloModel;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    let start = std::time::Instant::now();

    let model_path = Path::new("/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/models/v1/best_ir9_opset19.onnx"); // Convert your .pt to .onnx first
    let val_images_dir = Path::new("/Users/maximecolliat/PycharmProjects/PythonProject/ClashVision/data/images/val");
    let output_dir = Path::new("inference_results");

    // COCO class names (you should load this from your dataset config)
    let class_names = ClashClass::values()
        .iter()
        .map(|c| c.as_str().to_owned())
        .collect::<Vec<String>>();

    // Load the model
    println!("Loading model from: {:?}", model_path);
    let mut model = YoloModel::new(model_path, class_names)?;

    // Run inference on directory
    if val_images_dir.exists() {
        println!("Running inference on validation images...");
        let results_summary = model.run_inference_on_directory(val_images_dir, output_dir)?;

        // Print summary
        println!("\nInference completed! Results saved to: {:?}", output_dir);
        println!("Processed {} images", results_summary.len());

        let total_detections: usize = results_summary.iter().map(|r| r.num_detections).sum();
        println!("Total detections: {}", total_detections);

        // Print per-image summary
        for result in &results_summary {
            println!(
                "{}: {} detections",
                result.image_name, result.num_detections
            );
        }
    } else {
        println!(
            "Validation images directory not found: {:?}",
            val_images_dir
        );
        println!("Please add some images to test inference");
    }

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);

    Ok(())
}
