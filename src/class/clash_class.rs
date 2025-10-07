use std::fmt::Debug;

/// This file is part of a Clash of Clans related project.
#[derive(PartialEq, Eq)]
#[must_use]
pub enum ClashClass {
    ElixirStorage = 0,
    GoldStorage = 1,
}

impl ClashClass {
    /// Returns the string representation of the `ClashClass` variant.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ElixirStorage => "Elixir Storage",
            Self::GoldStorage => "Gold Storage",
        }
    }

    /// Returns a static slice of all `ClashClass` variants.
    pub fn values() -> &'static [Self] {
        static VALUES: [ClashClass; 2] = [ClashClass::ElixirStorage, ClashClass::GoldStorage];
        &VALUES
    }

    /// Returns the RGB color associated with the `ClashClass` variant.
    #[inline]
    #[must_use]
    pub const fn to_color(&self) -> (u8, u8, u8, u8) {
        match self {
            Self::ElixirStorage => (255, 0, 255, 255), // Magenta
            Self::GoldStorage => (212, 175, 55, 255),  // Gold
        }
    }

    /// Returns a static slice of RGB colors corresponding to the `ClashClass` variants.
    #[must_use]
    pub fn colors() -> &'static [(u8, u8, u8, u8)] {
        static COLORS: [(u8, u8, u8, u8); 2] = [
            (255, 0, 255, 255),  // Magenta for Elixir Storage
            (212, 175, 55, 255), // Gold for Gold Storage
        ];
        &COLORS
    }

    /// Returns the number of `ClashClass` variants.
    #[inline]
    #[must_use]
    pub fn num_classes() -> usize {
        Self::values().len()
    }
}

impl Debug for ClashClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::ClashClass;

    #[test]
    fn test_as_str() {
        assert_eq!(ClashClass::ElixirStorage.as_str(), "Elixir Storage");
        assert_eq!(ClashClass::GoldStorage.as_str(), "Gold Storage");
    }

    #[test]
    fn test_enum_values() {
        assert_eq!(ClashClass::ElixirStorage as usize, 0);
        assert_eq!(ClashClass::GoldStorage as usize, 1);
    }

    #[test]
    fn test_values() {
        let values = ClashClass::values();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], ClashClass::ElixirStorage);
        assert_eq!(values[1], ClashClass::GoldStorage);
    }

    #[test]
    fn test_colors() {
        assert_eq!(ClashClass::ElixirStorage.to_color(), (255, 0, 255, 255));
        assert_eq!(ClashClass::GoldStorage.to_color(), (212, 175, 55, 255));
    }

    #[test]
    fn test_colors_array() {
        let colors = ClashClass::colors();
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[0], (255, 0, 255, 255));
        assert_eq!(colors[1], (212, 175, 55, 255));
    }

    #[test]
    fn test_num_classes() {
        assert_eq!(ClashClass::num_classes(), 2);
    }
}
