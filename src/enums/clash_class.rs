use std::fmt::Debug;

/// This file is part of a Clash of Clans related project.
#[derive(PartialEq, Eq)]
pub enum ClashClass {
    ElixirStorage = 0,
    GoldStorage = 1,
}

impl ClashClass {
    /// Converts a usize to a ClashClass variant, if possible.
    pub fn from_usize(value: usize) -> Option<Self> {
        match value {
            0 => Some(ClashClass::ElixirStorage),
            1 => Some(ClashClass::GoldStorage),
            _ => None,
        }
    }

    /// Returns the string representation of the ClashClass variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            ClashClass::ElixirStorage => "Elixir Storage",
            ClashClass::GoldStorage => "Gold Storage",
        }
    }

    /// Returns a static slice of all ClashClass variants.
    pub fn values() -> &'static [ClashClass] {
        static VALUES: [ClashClass; 2] = [ClashClass::ElixirStorage, ClashClass::GoldStorage];
        &VALUES
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
    fn test_from_usize() {
        assert_eq!(ClashClass::from_usize(0), Some(ClashClass::ElixirStorage));
        assert_eq!(ClashClass::from_usize(1), Some(ClashClass::GoldStorage));
        assert_eq!(ClashClass::from_usize(2), None);
    }

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
}
