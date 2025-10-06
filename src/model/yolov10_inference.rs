use crate::detection::BoundingBox;
use crate::model::inference::YoloInference;
use ndarray::Array;

/// `YOLOv10` inference implementation
pub struct Yolov10Inference;

impl YoloInference for Yolov10Inference {
    fn parse_output(
        &self,
        output: &Array<f32, ndarray::IxDyn>,
        confidence_threshold: f32,
    ) -> Vec<BoundingBox> {
        let shape = output.shape();
        let reshaped_output = output
            .to_shape((shape[1], shape[2]))
            .expect("Failed to reshape YOLOv10 output");

        let mut boxes = Vec::with_capacity(reshaped_output.shape()[0]);

        for detection in reshaped_output.outer_iter() {
            let confidence = detection[4];

            if confidence >= confidence_threshold {
                let bbox = BoundingBox::new(
                    detection[0],
                    detection[1],
                    detection[2],
                    detection[3],
                    detection[5] as usize,
                    confidence,
                );
                boxes.push(bbox);
            }
        }

        boxes
    }
}
