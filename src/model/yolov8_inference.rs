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
        let num_rows = shape[1];
        let num_detections = shape[2];
        let num_classes = num_rows - 4;

        // Reshape to 2D view; fall back to a contiguous copy if the view is not standard-layout.
        let reshaped_view = output
            .to_shape((num_rows, num_detections))
            .expect("Failed to reshape YOLOv8 output");
        let owned_storage;
        let raw: &[f32] = if let Some(slice) = reshaped_view.as_slice() {
            slice
        } else {
            owned_storage = reshaped_view.as_standard_layout().to_owned();
            owned_storage.as_slice().expect("standard-layout array must be contiguous")
        };

        let stride = num_detections;
        let mut boxes = Vec::with_capacity(64);

        for det in 0..num_detections {
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
                boxes.push(BoundingBox::from_center(
                    x,
                    y,
                    w,
                    h,
                    max_class_id,
                    max_class_prob,
                ));
            }
        }

        boxes
    }
}
