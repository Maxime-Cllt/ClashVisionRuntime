use crate::image::image_size::ImageSize;
use ndarray::Array4;

/// A struct representing a loaded image with its pixel data and size.
#[derive(Clone)]
#[must_use]
#[non_exhaustive]
pub struct LoadedImage<T> {
    pub image_array: Array4<T>,
    pub size: ImageSize,
}

pub type LoadedImageU8 = LoadedImage<u8>;
pub type LoadedImageF32 = LoadedImage<f32>;

impl<T> LoadedImage<T> {
    /// Creates a new `LoadedImage`
    #[inline]
    pub const fn new(image_array: Array4<T>, size: ImageSize) -> Self {
        Self { image_array, size }
    }

    /// Returns the shape of the image array
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.image_array.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_loaded_image_creation() {
        let image_array = array![[[[1u8]], [[2u8]]], [[[3u8]], [[4u8]]]];
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let loaded_image = LoadedImage::new(image_array.clone(), size);

        assert_eq!(loaded_image.image_array, image_array);
        assert_eq!(loaded_image.size.width, size.width);
        assert_eq!(loaded_image.size.height, size.height);
    }

    #[test]
    fn test_loaded_image_shape() {
        let image_array = array![[[[1u8]], [[2u8]]], [[[3u8]], [[4u8]]]];
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let loaded_image = LoadedImage::new(image_array, size);

        assert_eq!(loaded_image.shape(), &[2, 2, 1, 1]);
    }
}
