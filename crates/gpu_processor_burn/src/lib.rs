//! The `burn_processor` crate.
// Intended for exploring GPU-accelerated image processing using the Burn framework.

pub mod backend;
pub mod kernel;
pub mod ops;
pub mod dataset;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
} 