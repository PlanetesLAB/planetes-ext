//! Interpolation on sorted grids.

/// Strategy for evaluating points outside the sample grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Extrapolate {
    /// Return the nearest endpoint value.
    #[default]
    Clamp,
    /// Linearly extrapolate using the two nearest samples.
    Linear,
}

/// Linearly interpolate `x` onto the grid defined by `xp` and `yp`.
///
/// Values outside the grid are clamped to the nearest endpoint value. Use
/// [`linear_with`] to instead extrapolate out-of-bounds values.
///
/// # Arguments
/// * `x` - The coordinate to evaluate.
/// * `xp` - The sample coordinates, sorted in ascending order.
/// * `yp` - The sample values, one per entry in `xp`.
///
/// # Panics
/// Panics if `xp` and `yp` have different lengths, or if `xp` is empty.
///
/// # Examples
/// ```
/// use planetes_ext::interp;
///
/// let xp = [0.0, 1.0, 2.0];
/// let yp = [10.0, 20.0, 40.0];
/// assert_eq!(interp::linear(0.5, &xp, &yp), 15.0);
/// assert_eq!(interp::linear(-1.0, &xp, &yp), 10.0); // clamped below
/// assert_eq!(interp::linear(3.0, &xp, &yp), 40.0); // clamped above
/// ```
#[must_use]
#[inline]
pub fn linear(x: f64, xp: &[f64], yp: &[f64]) -> f64 {
    linear_with(x, xp, yp, Extrapolate::Clamp)
}

/// Linearly interpolate `x` onto the grid defined by `xp` and `yp`, using an
/// explicit out-of-bounds [`Extrapolate`] strategy.
///
/// # Arguments
/// * `x` - The coordinate to evaluate.
/// * `xp` - The sample coordinates, sorted in ascending order.
/// * `yp` - The sample values, one per entry in `xp`.
/// * `extrapolate` - How to treat `x` outside `[xp[0], xp[n - 1]]`.
///
/// # Panics
/// Panics if `xp` and `yp` have different lengths, if `xp` is empty, or if
/// [`Extrapolate::Linear`] is requested with fewer than two points.
///
/// # Examples
/// ```
/// use planetes_ext::interp::{self, Extrapolate};
///
/// let xp = [0.0, 1.0, 2.0];
/// let yp = [10.0, 20.0, 40.0];
/// // Linearly extrapolate below the grid using the first two samples.
/// assert_eq!(interp::linear_with(-1.0, &xp, &yp, Extrapolate::Linear), 0.0);
/// ```
#[must_use]
#[inline]
pub fn linear_with(x: f64, xp: &[f64], yp: &[f64], extrapolate: Extrapolate) -> f64 {
    assert_eq!(xp.len(), yp.len(), "xp and yp must have the same length");
    assert!(!xp.is_empty(), "xp must not be empty");
    if matches!(extrapolate, Extrapolate::Linear) {
        assert!(
            xp.len() >= 2,
            "linear extrapolation requires at least two points"
        );
    }
    if x.is_nan() {
        return f64::NAN;
    }

    let n = xp.len();

    if x <= xp[0] {
        return match extrapolate {
            Extrapolate::Clamp => yp[0],
            Extrapolate::Linear => {
                yp[0] + (yp[1] - yp[0]) * (x - xp[0]) / (xp[1] - xp[0])
            }
        };
    }

    if x >= xp[n - 1] {
        return match extrapolate {
            Extrapolate::Clamp => yp[n - 1],
            Extrapolate::Linear => {
                yp[n - 1] + (yp[n - 1] - yp[n - 2]) * (x - xp[n - 1]) / (xp[n - 1] - xp[n - 2])
            }
        };
    }

    let i = xp.partition_point(|&v| v <= x);
    let x0 = xp[i - 1];
    let x1 = xp[i];
    let y0 = yp[i - 1];
    let y1 = yp[i];
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::{Extrapolate, linear, linear_with};

    #[test]
    fn test_linear_in_range() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        assert_eq!(linear(0.0, &xp, &yp), 10.0);
        assert_eq!(linear(0.5, &xp, &yp), 15.0);
        assert_eq!(linear(1.5, &xp, &yp), 30.0);
        assert_eq!(linear(2.0, &xp, &yp), 40.0);
    }

    #[test]
    fn test_linear_non_uniform_grid() {
        let xp = [0.0, 1.0, 4.0];
        let yp = [0.0, 1.0, 4.0];
        assert_eq!(linear(2.5, &xp, &yp), 2.5);
    }

    #[test]
    fn test_linear_below_range_clamps() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        assert_eq!(linear(-100.0, &xp, &yp), 10.0);
    }

    #[test]
    fn test_linear_above_range_clamps() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        assert_eq!(linear(100.0, &xp, &yp), 40.0);
    }

    #[test]
    fn test_linear_single_point() {
        let xp = [5.0];
        let yp = [42.0];
        assert_eq!(linear(5.0, &xp, &yp), 42.0);
        assert_eq!(linear(0.0, &xp, &yp), 42.0);
        assert_eq!(linear(100.0, &xp, &yp), 42.0);
    }

    #[test]
    fn test_linear_nan_propagates() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        assert!(linear(f64::NAN, &xp, &yp).is_nan());
    }

    #[test]
    #[should_panic(expected = "xp and yp must have the same length")]
    fn test_linear_mismatched_lengths_panics() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0];
        let _ = linear(0.5, &xp, &yp);
    }

    #[test]
    #[should_panic(expected = "xp must not be empty")]
    fn test_linear_empty_panics() {
        let xp: [f64; 0] = [];
        let yp: [f64; 0] = [];
        let _ = linear(0.5, &xp, &yp);
    }

    #[test]
    fn test_linear_with_extrapolate_below() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        // First-segment slope is 10; y(-1) = 10 + 10 * (-1) = 0.
        assert_eq!(linear_with(-1.0, &xp, &yp, Extrapolate::Linear), 0.0);
    }

    #[test]
    fn test_linear_with_extrapolate_above() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        // Last-segment slope is 20; y(3) = 40 + 20 * 1 = 60.
        assert_eq!(linear_with(3.0, &xp, &yp, Extrapolate::Linear), 60.0);
    }

    #[test]
    fn test_linear_with_clamp_matches_linear() {
        let xp = [0.0, 1.0, 2.0];
        let yp = [10.0, 20.0, 40.0];
        for &x in &[-1.0, 0.0, 0.5, 1.5, 2.0, 3.0] {
            assert_eq!(linear_with(x, &xp, &yp, Extrapolate::Clamp), linear(x, &xp, &yp));
        }
    }

    #[test]
    #[should_panic(expected = "linear extrapolation requires at least two points")]
    fn test_linear_with_extrapolate_single_point_panics() {
        let xp = [5.0];
        let yp = [42.0];
        let _ = linear_with(0.0, &xp, &yp, Extrapolate::Linear);
    }
}
