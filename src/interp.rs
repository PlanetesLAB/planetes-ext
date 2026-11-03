//! Interpolation on sorted grids.

use ndarray::ArrayView2;

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
            Extrapolate::Linear => yp[0] + (yp[1] - yp[0]) * (x - xp[0]) / (xp[1] - xp[0]),
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

/// Bilinearly interpolate `(x, y)` onto the grid defined by `xp`, `yp` and `zp`.
///
/// `zp` has shape `(yp.len(), xp.len())`, where `zp[[iy, ix]]` holds the value
/// at `(xp[ix], yp[iy])`. Points outside the grid are clamped to the nearest
/// edge before interpolation; use [`bilinear_with`] to instead extrapolate.
///
/// # Panics
/// Panics if `zp` does not have shape `(yp.len(), xp.len())`, or if `xp` or `yp`
/// has fewer than two points.
///
/// # Examples
/// ```
/// use ndarray::array;
/// use planetes_ext::interp;
///
/// let xp = [0.0, 1.0];
/// let yp = [0.0, 1.0];
/// let zp = array![[0.0, 0.0], [0.0, 1.0]]; // f(x, y) = x * y
/// assert_eq!(interp::bilinear(0.5, 0.5, &xp, &yp, zp.view()), 0.25);
/// assert_eq!(interp::bilinear(2.0, -1.0, &xp, &yp, zp.view()), 0.0); // clamped
/// ```
#[must_use]
#[inline]
pub fn bilinear(x: f64, y: f64, xp: &[f64], yp: &[f64], zp: ArrayView2<f64>) -> f64 {
    bilinear_with(x, y, xp, yp, zp, Extrapolate::Clamp)
}

/// Bilinearly interpolate `(x, y)` onto the grid defined by `xp`, `yp` and `zp`,
/// using an explicit out-of-bounds [`Extrapolate`] strategy.
///
/// `zp` has shape `(yp.len(), xp.len())`, where `zp[[iy, ix]]` holds the value
/// at `(xp[ix], yp[iy])`.
///
/// # Panics
/// Panics if `zp` does not have shape `(yp.len(), xp.len())`, or if `xp` or `yp`
/// has fewer than two points.
///
/// # Examples
/// ```
/// use ndarray::array;
/// use planetes_ext::interp::{self, Extrapolate};
///
/// let xp = [0.0, 1.0];
/// let yp = [0.0, 1.0];
/// let zp = array![[0.0, 0.0], [0.0, 1.0]]; // f(x, y) = x * y
/// assert_eq!(interp::bilinear_with(2.0, 3.0, &xp, &yp, zp.view(), Extrapolate::Linear), 6.0);
/// ```
#[must_use]
#[inline]
pub fn bilinear_with(
    x: f64,
    y: f64,
    xp: &[f64],
    yp: &[f64],
    zp: ArrayView2<f64>,
    extrapolate: Extrapolate,
) -> f64 {
    assert_eq!(
        zp.dim(),
        (yp.len(), xp.len()),
        "zp must have shape (yp.len(), xp.len())"
    );
    assert!(xp.len() >= 2, "xp must have at least two points");
    assert!(yp.len() >= 2, "yp must have at least two points");
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    let (xe, ye) = match extrapolate {
        Extrapolate::Clamp => (
            x.clamp(xp[0], xp[xp.len() - 1]),
            y.clamp(yp[0], yp[yp.len() - 1]),
        ),
        Extrapolate::Linear => (x, y),
    };

    let ix = cell_index(xe, xp);
    let iy = cell_index(ye, yp);

    let x0 = xp[ix];
    let x1 = xp[ix + 1];
    let y0 = yp[iy];
    let y1 = yp[iy + 1];

    let f00 = zp[[iy, ix]];
    let f10 = zp[[iy, ix + 1]];
    let f01 = zp[[iy + 1, ix]];
    let f11 = zp[[iy + 1, ix + 1]];

    let dx = (xe - x0) / (x1 - x0);
    let dy = (ye - y0) / (y1 - y0);

    let top = f00 + (f10 - f00) * dx;
    let bot = f01 + (f11 - f01) * dx;
    top + (bot - top) * dy
}

/// Returns the left index of the cell containing `v`, clamped to `[0, n - 2]`.
///
/// `grid` must be sorted ascending with at least two entries.
#[inline]
fn cell_index(v: f64, grid: &[f64]) -> usize {
    if v <= grid[0] {
        0
    } else if v >= grid[grid.len() - 1] {
        grid.len() - 2
    } else {
        grid.partition_point(|&g| g <= v) - 1
    }
}

/// Error returned when a cubic spline cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineError {
    /// The x and y samples have different lengths.
    DifferentLengths,
    /// Fewer than two sample points were provided.
    InsufficientPoints,
    /// The x samples are not in strictly ascending order.
    UnsortedPoints,
    /// A numerically unstable pivot was encountered during construction.
    NumericalError,
}

impl std::fmt::Display for SplineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplineError::DifferentLengths => write!(f, "x and y must have the same length"),
            SplineError::InsufficientPoints => write!(f, "at least two points are required"),
            SplineError::UnsortedPoints => write!(f, "x must be in strictly ascending order"),
            SplineError::NumericalError => write!(f, "numerically unstable spline construction"),
        }
    }
}

impl std::error::Error for SplineError {}

/// Natural cubic spline interpolator over a 1-D grid.
///
/// The spline is built once from sample points and then evaluated many times.
/// The second derivative vanishes at the endpoints (the "natural" boundary
/// condition). Out-of-bounds evaluations are clamped to the nearest endpoint
/// value; use [`eval_with`](CubicSpline::eval_with) to extrapolate instead.
///
/// # Examples
/// ```
/// use planetes_ext::interp::CubicSpline;
///
/// let x = [0.0, 1.0, 2.0, 3.0];
/// let y = [0.0, 2.0, 4.0, 6.0];
/// let spline = CubicSpline::new(&x, &y).unwrap();
/// assert_eq!(spline.eval(1.5), 3.0);
/// ```
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    y2: Vec<f64>,
}

impl CubicSpline {
    /// Constructs a natural cubic spline from sample points `(x, y)`.
    ///
    /// # Errors
    /// Returns an error if `x` and `y` differ in length, if fewer than two
    /// points are given, if `x` is not strictly ascending, or if construction
    /// is numerically unstable.
    pub fn new(x: &[f64], y: &[f64]) -> Result<CubicSpline, SplineError> {
        if x.len() != y.len() {
            return Err(SplineError::DifferentLengths);
        }
        let n = x.len();
        if n < 2 {
            return Err(SplineError::InsufficientPoints);
        }
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err(SplineError::UnsortedPoints);
            }
        }

        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n - 1];

        for i in 1..n - 1 {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            if p.abs() < f64::EPSILON {
                return Err(SplineError::NumericalError);
            }
            y2[i] = (sig - 1.0) / p;
            u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        }

        for i in (0..n - 1).rev() {
            y2[i] = y2[i] * y2[i + 1] + u[i];
        }

        Ok(CubicSpline {
            x: x.to_vec(),
            y: y.to_vec(),
            y2,
        })
    }

    /// Evaluates the spline at `x`.
    ///
    /// Out-of-bounds values are clamped to the nearest endpoint value. Use
    /// [`eval_with`](CubicSpline::eval_with) to extrapolate instead.
    #[must_use]
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        self.eval_with(x, Extrapolate::Clamp)
    }

    /// Evaluates the spline at `x` with an explicit [`Extrapolate`] strategy.
    ///
    /// # Examples
    /// ```
    /// use planetes_ext::interp::{CubicSpline, Extrapolate};
    ///
    /// let x = [0.0, 1.0, 2.0];
    /// let y = [0.0, 2.0, 4.0];
    /// let spline = CubicSpline::new(&x, &y).unwrap();
    /// assert_eq!(spline.eval_with(-1.0, Extrapolate::Clamp), 0.0);
    /// assert_eq!(spline.eval_with(3.0, Extrapolate::Linear), 6.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn eval_with(&self, x: f64, extrapolate: Extrapolate) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        let n = self.x.len();

        if x <= self.x[0] {
            return match extrapolate {
                Extrapolate::Clamp => self.y[0],
                Extrapolate::Linear => {
                    self.y[0] + (self.y[1] - self.y[0]) * (x - self.x[0]) / (self.x[1] - self.x[0])
                }
            };
        }
        if x >= self.x[n - 1] {
            return match extrapolate {
                Extrapolate::Clamp => self.y[n - 1],
                Extrapolate::Linear => {
                    self.y[n - 1]
                        + (self.y[n - 1] - self.y[n - 2]) * (x - self.x[n - 1])
                            / (self.x[n - 1] - self.x[n - 2])
                }
            };
        }

        let k = cell_index(x, &self.x);
        let h = self.x[k + 1] - self.x[k];
        let a = (self.x[k + 1] - x) / h;
        let b = (x - self.x[k]) / h;
        a * self.y[k]
            + b * self.y[k + 1]
            + ((a * a * a - a) * self.y2[k] + (b * b * b - b) * self.y2[k + 1]) * (h * h) / 6.0
    }
}

/// How to extrapolate values outside the original grid during remapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMode {
    /// Out-of-bounds input is an error.
    Strict,
    /// Return the boundary value.
    Constant,
    /// Extrapolate with a log-log power law through the two boundary samples.
    PowerLaw,
    /// Return zero.
    Zero,
    /// Taper linearly to zero over one cell beyond the boundary.
    LinearDrop,
}

/// Error returned when remapping fails.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RemapError {
    /// `x` fell outside the grid and the relevant bound is `Strict`.
    OutOfBounds(f64),
    /// The values and the grid have different lengths.
    MismatchedLength,
    /// The original grid is empty.
    EmptyGrid,
}

impl std::fmt::Display for RemapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemapError::OutOfBounds(x) => write!(f, "x = {x} is out of bounds of the grid"),
            RemapError::MismatchedLength => write!(f, "values and grid have different lengths"),
            RemapError::EmptyGrid => write!(f, "grid is empty"),
        }
    }
}

impl std::error::Error for RemapError {}

/// Builder that remaps values onto a new grid.
///
/// Constructed via [`Remap::remap_from`]; configure the out-of-bounds behavior
/// with [`with_lower_bound`](RemapBuilder::with_lower_bound) and
/// [`with_upper_bound`](RemapBuilder::with_upper_bound), then finish with
/// [`onto_grid`](RemapBuilder::onto_grid).
#[must_use]
pub struct RemapBuilder<'a> {
    grid: &'a [f64],
    values: &'a [f64],
    lower: ExtrapolationMode,
    upper: ExtrapolationMode,
}

impl<'a> RemapBuilder<'a> {
    fn new(grid: &'a [f64], values: &'a [f64]) -> Self {
        Self {
            grid,
            values,
            lower: ExtrapolationMode::Strict,
            upper: ExtrapolationMode::Strict,
        }
    }

    /// Sets the behavior when `x_new` is below the lowest grid value.
    pub const fn with_lower_bound(mut self, mode: ExtrapolationMode) -> Self {
        self.lower = mode;
        self
    }

    /// Sets the behavior when `x_new` is above the highest grid value.
    pub const fn with_upper_bound(mut self, mode: ExtrapolationMode) -> Self {
        self.upper = mode;
        self
    }

    /// Remaps the values onto `x_new`, returning one value per entry.
    ///
    /// # Errors
    /// Returns an error if the values and grid differ in length, if the grid is
    /// empty, or if a point falls outside the grid and the relevant bound is
    /// [`ExtrapolationMode::Strict`].
    pub fn onto_grid(self, x_new: &[f64]) -> Result<Vec<f64>, RemapError> {
        if self.values.len() != self.grid.len() {
            return Err(RemapError::MismatchedLength);
        }
        let n = self.grid.len();
        if n == 0 {
            return Err(RemapError::EmptyGrid);
        }
        let grid = self.grid;
        let values = self.values;

        let mut out = Vec::with_capacity(x_new.len());

        if n == 1 {
            let (x0, f0) = (grid[0], values[0]);
            for &x in x_new {
                if x < x0 {
                    out.push(Self::extrapolate(self.lower, x, x0, f0, None)?);
                } else if x > x0 {
                    out.push(Self::extrapolate(self.upper, x, x0, f0, None)?);
                } else {
                    out.push(f0);
                }
            }
            return Ok(out);
        }

        let x_lo = grid[0];
        let x_hi = grid[n - 1];
        let f_lo = values[0];
        let f_hi = values[n - 1];
        let low_neighbor = Some((grid[1], values[1]));
        let high_neighbor = Some((grid[n - 2], values[n - 2]));

        for &x in x_new {
            if x < x_lo {
                out.push(Self::extrapolate(self.lower, x, x_lo, f_lo, low_neighbor)?);
                continue;
            }
            if x > x_hi {
                out.push(Self::extrapolate(self.upper, x, x_hi, f_hi, high_neighbor)?);
                continue;
            }

            let i = cell_index(x, grid);
            let x0 = grid[i];
            let x1 = grid[i + 1];
            let f0 = values[i];
            let f1 = values[i + 1];
            let eps = (x - x0) / (x1 - x0);
            out.push(Self::interp_pair(eps, f0, f1));
        }

        Ok(out)
    }

    /// Log-space linear interpolation, falling back to linear when either
    /// endpoint is non-positive.
    #[inline]
    fn interp_pair(eps: f64, f0: f64, f1: f64) -> f64 {
        if f0 > 0.0 && f1 > 0.0 {
            ((1.0 - eps) * f0.ln() + eps * f1.ln()).exp()
        } else {
            (1.0 - eps) * f0 + eps * f1
        }
    }

    #[inline]
    fn extrapolate(
        mode: ExtrapolationMode,
        x: f64,
        x_b: f64,
        f_b: f64,
        neighbor: Option<(f64, f64)>,
    ) -> Result<f64, RemapError> {
        match mode {
            ExtrapolationMode::Strict => Err(RemapError::OutOfBounds(x)),
            ExtrapolationMode::Constant => Ok(f_b),
            ExtrapolationMode::Zero => Ok(0.0),
            ExtrapolationMode::PowerLaw => {
                let Some((x_n, f_n)) = neighbor else {
                    return Ok(f_b);
                };
                if f_n > 0.0 && f_b > 0.0 {
                    let ratio = f_n / f_b;
                    let exponent = (x.ln() - x_b.ln()) / (x_n.ln() - x_b.ln());
                    Ok(f_b * ratio.powf(exponent))
                } else {
                    Ok(0.0)
                }
            }
            ExtrapolationMode::LinearDrop => {
                let Some((x_n, _)) = neighbor else {
                    return Ok(f_b);
                };
                let eps = (x - x_b) / (x_n - x_b);
                if eps > -1.0 {
                    Ok((1.0 - eps.abs()) * f_b)
                } else {
                    Ok(0.0)
                }
            }
        }
    }
}

/// Trait for remapping values from one grid onto another.
///
/// # Examples
/// ```
/// use planetes_ext::interp::{ExtrapolationMode, Remap};
///
/// let grid = [1.0, 2.0, 4.0];
/// let values = [10.0, 40.0, 160.0];
/// let remapped = values
///     .remap_from(&grid)
///     .with_lower_bound(ExtrapolationMode::Constant)
///     .with_upper_bound(ExtrapolationMode::Constant)
///     .onto_grid(&[0.5, 3.0, 8.0])
///     .unwrap();
/// // 0.5 and 8.0 clamp to the boundary values; 3.0 is interpolated in log space.
/// assert_eq!(remapped[0], 10.0);
/// assert!((remapped[1] - 80.0).abs() < 1e-9);
/// assert_eq!(remapped[2], 160.0);
/// ```
pub trait Remap {
    /// Returns a builder that remaps `self` from `original_grid`.
    fn remap_from<'a>(&'a self, original_grid: &'a [f64]) -> RemapBuilder<'a>;
}

impl Remap for [f64] {
    fn remap_from<'a>(&'a self, original_grid: &'a [f64]) -> RemapBuilder<'a> {
        RemapBuilder::new(original_grid, self)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::CubicSpline;
    use super::Remap;
    use super::{Extrapolate, ExtrapolationMode, RemapError, SplineError};
    use super::{bilinear, bilinear_with, linear, linear_with};

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
            assert_eq!(
                linear_with(x, &xp, &yp, Extrapolate::Clamp),
                linear(x, &xp, &yp)
            );
        }
    }

    #[test]
    #[should_panic(expected = "linear extrapolation requires at least two points")]
    fn test_linear_with_extrapolate_single_point_panics() {
        let xp = [5.0];
        let yp = [42.0];
        let _ = linear_with(0.0, &xp, &yp, Extrapolate::Linear);
    }

    // ── bilinear ──────────────────────────────────────────────────────

    #[test]
    fn test_bilinear_interior() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 2.0], [1.0, 3.0]]; // f(x, y) = 2x + y
        assert_eq!(bilinear(0.5, 0.5, &xp, &yp, zp.view()), 1.5);
        assert_eq!(bilinear(0.25, 0.75, &xp, &yp, zp.view()), 1.25);
    }

    #[test]
    fn test_bilinear_cross_term() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 0.0], [0.0, 1.0]]; // f(x, y) = x * y
        assert_eq!(bilinear(0.5, 0.5, &xp, &yp, zp.view()), 0.25);
        assert_eq!(bilinear(0.5, 0.25, &xp, &yp, zp.view()), 0.125);
    }

    #[test]
    fn test_bilinear_clamps_out_of_bounds() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 2.0], [1.0, 3.0]]; // f(x, y) = 2x + y
        assert_eq!(bilinear(5.0, 5.0, &xp, &yp, zp.view()), 3.0); // (1, 1)
        assert_eq!(bilinear(-5.0, -5.0, &xp, &yp, zp.view()), 0.0); // (0, 0)
        assert_eq!(bilinear(5.0, 0.0, &xp, &yp, zp.view()), 2.0); // (1, 0)
    }

    #[test]
    fn test_bilinear_non_uniform_grid() {
        let xp = [0.0, 1.0, 3.0];
        let yp = [0.0, 2.0];
        let zp = array![[0.0, 2.0, 6.0], [2.0, 4.0, 8.0]]; // f(x, y) = 2x + y
        assert_eq!(bilinear(2.0, 1.0, &xp, &yp, zp.view()), 5.0);
    }

    #[test]
    fn test_bilinear_with_extrapolate_linear() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 0.0], [0.0, 1.0]]; // f(x, y) = x * y
        assert_eq!(
            bilinear_with(2.0, 3.0, &xp, &yp, zp.view(), Extrapolate::Linear),
            6.0
        );
    }

    #[test]
    fn test_bilinear_nan_propagates() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 0.0], [0.0, 1.0]];
        assert!(bilinear(f64::NAN, 0.5, &xp, &yp, zp.view()).is_nan());
        assert!(bilinear(0.5, f64::NAN, &xp, &yp, zp.view()).is_nan());
    }

    #[test]
    #[should_panic(expected = "zp must have shape")]
    fn test_bilinear_shape_mismatch_panics() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0, 2.0];
        let zp = array![[0.0, 0.0], [0.0, 1.0]]; // shape (2, 2), expected (3, 2)
        let _ = bilinear(0.5, 0.5, &xp, &yp, zp.view());
    }

    #[test]
    #[should_panic(expected = "xp must have at least two points")]
    fn test_bilinear_too_few_points_panics() {
        let xp = [0.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0], [1.0]]; // shape (2, 1)
        let _ = bilinear(0.5, 0.5, &xp, &yp, zp.view());
    }

    #[test]
    fn test_bilinear_with_clamp_matches_bilinear() {
        let xp = [0.0, 1.0];
        let yp = [0.0, 1.0];
        let zp = array![[0.0, 2.0], [1.0, 3.0]];
        for &x in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
            for &y in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
                assert_eq!(
                    bilinear_with(x, y, &xp, &yp, zp.view(), Extrapolate::Clamp),
                    bilinear(x, y, &xp, &yp, zp.view())
                );
            }
        }
    }

    // ── cubic spline ──────────────────────────────────────────────────

    #[test]
    fn test_cubic_spline_interpolates_nodes() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        assert_eq!(spline.eval(0.0), 1.0);
        assert_eq!(spline.eval(1.0), 3.0);
        assert_eq!(spline.eval(2.0), 2.0);
    }

    #[test]
    fn test_cubic_spline_interior_values() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        // Hand-derived natural-spline values (y2 = [0, -4.5, 0]).
        assert_eq!(spline.eval(0.5), 2.28125);
        assert_eq!(spline.eval(1.5), 2.78125);
    }

    #[test]
    fn test_cubic_spline_linear_data_is_exact() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 2.0, 4.0, 6.0]; // y = 2x
        let spline = CubicSpline::new(&x, &y).unwrap();
        assert_eq!(spline.eval(0.5), 1.0);
        assert_eq!(spline.eval(1.25), 2.5);
        assert_eq!(spline.eval(2.75), 5.5);
    }

    #[test]
    fn test_cubic_spline_extrapolates_linear() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        // Below: slope (3 - 1) / 1 = 2 -> y(-1) = 1 - 2 = -1.
        assert_eq!(spline.eval_with(-1.0, Extrapolate::Linear), -1.0);
        // Above: slope (2 - 3) / 1 = -1 -> y(3) = 2 - 1 = 1.
        assert_eq!(spline.eval_with(3.0, Extrapolate::Linear), 1.0);
    }

    #[test]
    fn test_cubic_spline_clamps() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        assert_eq!(spline.eval_with(-1.0, Extrapolate::Clamp), 1.0);
        assert_eq!(spline.eval_with(3.0, Extrapolate::Clamp), 2.0);
    }

    #[test]
    fn test_cubic_spline_eval_defaults_to_clamp() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        // Out of bounds, `eval` clamps to the endpoint value.
        assert_eq!(spline.eval(-1.0), 1.0);
        assert_eq!(spline.eval(3.0), 2.0);
        // In bounds, `eval` matches both explicit modes.
        assert_eq!(spline.eval(0.5), spline.eval_with(0.5, Extrapolate::Clamp));
        assert_eq!(spline.eval(0.5), spline.eval_with(0.5, Extrapolate::Linear));
    }

    #[test]
    fn test_cubic_spline_nan_propagates() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 3.0, 2.0];
        let spline = CubicSpline::new(&x, &y).unwrap();
        assert!(spline.eval(f64::NAN).is_nan());
    }

    #[test]
    fn test_cubic_spline_errors() {
        assert!(matches!(
            CubicSpline::new(&[0.0, 1.0], &[1.0]),
            Err(SplineError::DifferentLengths)
        ));
        assert!(matches!(
            CubicSpline::new(&[0.0], &[1.0]),
            Err(SplineError::InsufficientPoints)
        ));
        assert!(matches!(
            CubicSpline::new(&[0.0, 2.0, 1.0], &[1.0, 2.0, 3.0]),
            Err(SplineError::UnsortedPoints)
        ));
    }

    // ── remap ─────────────────────────────────────────────────────────

    #[test]
    fn test_remap_interpolation_geometric() {
        let grid = [1.0, 2.0];
        let values = [10.0, 1000.0];
        let out = values.remap_from(&grid).onto_grid(&[1.5]).unwrap();
        // Geometric mean: sqrt(10 * 1000) = 100.
        assert!((out[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_remap_linear_fallback() {
        let grid = [0.0, 1.0];
        let values = [-1.0, 1.0];
        let out = values.remap_from(&grid).onto_grid(&[0.5]).unwrap();
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn test_remap_extrapolation_modes_below() {
        let grid = [1.0, 2.0, 4.0];
        let values = [10.0, 40.0, 160.0];
        let below = [0.5];

        let constant = values
            .remap_from(&grid)
            .with_lower_bound(ExtrapolationMode::Constant)
            .onto_grid(&below)
            .unwrap();
        assert_eq!(constant, vec![10.0]);

        let zero = values
            .remap_from(&grid)
            .with_lower_bound(ExtrapolationMode::Zero)
            .onto_grid(&below)
            .unwrap();
        assert_eq!(zero, vec![0.0]);

        let drop = values
            .remap_from(&grid)
            .with_lower_bound(ExtrapolationMode::LinearDrop)
            .onto_grid(&below)
            .unwrap();
        // eps = (0.5 - 1) / (2 - 1) = -0.5 -> (1 - 0.5) * 10 = 5.
        assert_eq!(drop, vec![5.0]);

        let power = values
            .remap_from(&grid)
            .with_lower_bound(ExtrapolationMode::PowerLaw)
            .onto_grid(&below)
            .unwrap();
        // 10 * (40 / 10)^(ln(0.5) / ln(2)) = 10 * 4^-1 = 2.5.
        assert!((power[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_remap_upper_bound() {
        let grid = [1.0, 2.0, 4.0];
        let values = [10.0, 40.0, 160.0];
        let above = [8.0];
        let constant = values
            .remap_from(&grid)
            .with_upper_bound(ExtrapolationMode::Constant)
            .onto_grid(&above)
            .unwrap();
        assert_eq!(constant, vec![160.0]);
    }

    #[test]
    fn test_remap_strict_errors() {
        let grid = [1.0, 2.0, 4.0];
        let values = [10.0, 40.0, 160.0];
        let result = values.remap_from(&grid).onto_grid(&[0.5]);
        assert_eq!(result, Err(RemapError::OutOfBounds(0.5)));
    }

    #[test]
    fn test_remap_errors() {
        assert_eq!(
            [10.0, 40.0].remap_from(&[1.0, 2.0, 4.0]).onto_grid(&[1.0]),
            Err(RemapError::MismatchedLength)
        );
        let empty: [f64; 0] = [];
        assert_eq!(
            empty.remap_from(&[]).onto_grid(&[1.0]),
            Err(RemapError::EmptyGrid)
        );
    }

    #[test]
    fn test_remap_single_point() {
        let grid = [5.0];
        let values = [42.0];
        let out = values
            .remap_from(&grid)
            .with_lower_bound(ExtrapolationMode::Constant)
            .with_upper_bound(ExtrapolationMode::Zero)
            .onto_grid(&[0.0, 5.0, 10.0])
            .unwrap();
        assert_eq!(out, vec![42.0, 42.0, 0.0]);
    }
}
