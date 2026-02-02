use crate::detection::BoundingBox;
use crate::model::inference::YoloInference;
use ndarray::ArrayViewD;

/// `YOLOv8` inference implementation
pub struct Yolov8Inference;

impl YoloInference for Yolov8Inference {
    fn parse_output(
        &self,
        output: ArrayViewD<'_, f32>,
        confidence_threshold: f32,
    ) -> Vec<BoundingBox> {
        let shape = output.shape();
        let reshaped_output = output
            .to_shape((shape[1], shape[2]))
            .expect("Failed to reshape YOLOv8 output");

        let num_rows = reshaped_output.shape()[0];
        let num_detections = reshaped_output.shape()[1];
        let num_classes = num_rows - 4;

        let mut boxes = Vec::with_capacity(num_detections / 10);

        // Get raw slice for faster access (avoids per-element bounds checking)
        let raw = reshaped_output.as_slice().unwrap();
        let stride = num_detections; // row-major stride for (num_rows, num_detections) layout

        for det in 0..num_detections {
            // Find max class probability with a direct loop (no iterator overhead)
            let mut max_class_id = 0usize;
            let mut max_class_prob = raw[4 * stride + det];

            for c in 1..num_classes {
                let prob = raw[(4 + c) * stride + det];
                if prob > max_class_prob {
                    max_class_prob = prob;
                    max_class_id = c;
                }
            }

            if max_class_prob > confidence_threshold {
                let x = raw[det];
                let y = raw[stride + det];
                let w = raw[2 * stride + det];
                let h = raw[3 * stride + det];
                boxes.push(BoundingBox::from_center(x, y, w, h, max_class_id, max_class_prob));
            }
        }

        boxes
    }
}
