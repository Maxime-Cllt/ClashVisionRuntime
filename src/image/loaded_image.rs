use crate::image::image_size::ImageSize;
use ndarray::Array4;

#[derive(Clone)]
pub struct LoadedImage<T> {
    pub image_array: Array4<T>,
    pub size: ImageSize,
}

pub type LoadedImageU8 = LoadedImage<u8>;
pub type LoadedImageF32 = LoadedImage<f32>;

impl<T> LoadedImage<T> {
    pub fn new(image_array: Array4<T>, size: ImageSize) -> Self {
        Self { image_array, size }
    }

    pub fn shape(&self) -> &[usize] {
        self.image_array.shape()
    }
}
