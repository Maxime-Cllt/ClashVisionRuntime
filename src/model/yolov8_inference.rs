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

        // Class-major scan: for each class row, update the running best (prob, id) of every
        // detection. Contiguous reads + branchless select, so the compiler can vectorize it
        // (the detection-major loop strided by `num_detections` on every access).
        let mut best_prob: Vec<f32> = raw[4 * stride..5 * stride].to_vec();
        let mut best_class: Vec<u16> = vec![0; stride];
        for c in 1..num_classes {
            let row = &raw[(4 + c) * stride..(5 + c) * stride];
            for ((prob, class), &value) in best_prob.iter_mut().zip(best_class.iter_mut()).zip(row)
            {
                if value > *prob {
                    *prob = value;
                    *class = c as u16;
                }
            }
        }

        let mut boxes = Vec::with_capacity(64);
        for det in 0..num_detections {
            if best_prob[det] > confidence_threshold {
                boxes.push(BoundingBox::from_center(
                    raw[det],
                    raw[stride + det],
                    raw[2 * stride + det],
                    raw[3 * stride + det],
                    usize::from(best_class[det]),
                    best_prob[det],
                ));
            }
        }

        boxes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Straightforward detection-major reference implementation.
    fn reference(raw: &Array3<f32>, threshold: f32) -> Vec<(usize, f32)> {
        let (_, rows, dets) = raw.dim();
        let mut out = Vec::new();
        for d in 0..dets {
            let (mut id, mut prob) = (0, raw[[0, 4, d]]);
            for c in 1..rows - 4 {
                if raw[[0, 4 + c, d]] > prob {
                    prob = raw[[0, 4 + c, d]];
                    id = c;
                }
            }
            if prob > threshold {
                out.push((id, prob));
            }
        }
        out
    }

    #[test]
    fn class_major_scan_matches_reference() {
        let (classes, dets) = (7, 50);
        let raw = Array3::from_shape_fn((1, 4 + classes, dets), |(_, r, d)| {
            // deterministic pseudo-random values, with ties between classes
            (((r * 31 + d * 17) % 23) as f32) / 23.0
        });

        let boxes = Yolov8Inference.parse_output(raw.view().into_dyn(), 0.5);
        let got: Vec<(usize, f32)> = boxes.iter().map(|b| (b.class_id, b.confidence)).collect();
        assert_eq!(got, reference(&raw, 0.5));
    }
}
