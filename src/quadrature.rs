//! Numerical quadrature rules.

// Quadrature orders are far below 2^52, so `usize` -> `f64` is exact here.
#![allow(clippy::cast_precision_loss)]

use std::f64::consts::PI;

use crate::types::RVector;

/// Evaluates the Legendre polynomial `Pₙ` and its derivative at `z`.
#[inline]
fn legendre_pair(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = 1.0;
    let mut p2 = 0.0;
    for j in 0..n {
        let jf = j as f64;
        let p3 = p2;
        p2 = p1;
        p1 = ((2.0 * jf + 1.0) * z * p2 - jf * p3) / (jf + 1.0);
    }
    let dp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
    (p1, dp)
}

/// Returns the nodes and weights of the `n`-point Gauss–Legendre rule on
/// `[x1, x2]`.
///
/// The nodes are ordered from `x1` towards `x2`. An `n`-point rule integrates
/// polynomials up to degree `2n - 1` exactly. For `n == 0` both returned
/// vectors are empty.
///
/// # Examples
/// ```
/// use planetes_ext::quadrature::gauss_legendre;
///
/// // Two-point rule on [-1, 1].
/// let (x, w) = gauss_legendre(-1.0, 1.0, 2);
/// let sqrt_third = 1.0 / 3.0_f64.sqrt();
/// assert!((x[0] + sqrt_third).abs() < 1e-14);
/// assert!((x[1] - sqrt_third).abs() < 1e-14);
/// assert!((w[0] - 1.0).abs() < 1e-14);
/// assert!((w[1] - 1.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn gauss_legendre(x1: f64, x2: f64, n: usize) -> (RVector, RVector) {
    const EPS: f64 = 1e-15;
    const MAX_ITER: usize = 100;

    let m = n.div_ceil(2);
    let xm = f64::midpoint(x2, x1);
    let xl = 0.5 * (x2 - x1);
    let nf = n as f64;

    let mut x = RVector::zeros(n);
    let mut w = RVector::zeros(n);

    for i in 0..m {
        // Initial guess for the i-th root, refined by Newton iteration.
        let mut z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();

        for _ in 0..MAX_ITER {
            let (p, dp) = legendre_pair(n, z);
            let z_new = z - p / dp;
            let converged = (z_new - z).abs() <= EPS;
            z = z_new;
            if converged {
                break;
            }
        }

        let (_, dp) = legendre_pair(n, z);
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * dp * dp);
        w[n - 1 - i] = w[i];
    }

    (x, w)
}

#[cfg(test)]
mod tests {
    use super::gauss_legendre;

    fn integrate<F: Fn(f64) -> f64>(x1: f64, x2: f64, n: usize, f: F) -> f64 {
        let (x, w) = gauss_legendre(x1, x2, n);
        x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * f(xi)).sum()
    }

    #[test]
    fn test_empty_rule() {
        let (x, w) = gauss_legendre(0.0, 1.0, 0);
        assert!(x.is_empty());
        assert!(w.is_empty());
    }

    #[test]
    fn test_one_point() {
        let (x, w) = gauss_legendre(0.0, 10.0, 1);
        assert!((x[0] - 5.0).abs() < 1e-14);
        assert!((w[0] - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_two_point() {
        let (x, w) = gauss_legendre(-1.0, 1.0, 2);
        let sqrt_third = 1.0 / 3.0_f64.sqrt();
        assert!((x[0] + sqrt_third).abs() < 1e-14);
        assert!((x[1] - sqrt_third).abs() < 1e-14);
        assert!((w[0] - 1.0).abs() < 1e-14);
        assert!((w[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_nodes_are_sorted_and_symmetric() {
        let (x, w) = gauss_legendre(-2.0, 3.0, 17);
        let midpoint = 0.5;
        for i in 0..x.len() {
            if i > 0 {
                assert!(x[i] > x[i - 1]);
            }
            assert!((x[i] + x[x.len() - 1 - i] - 2.0 * midpoint).abs() < 1e-12);
            assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_weights_sum_to_interval_length() {
        for n in [1, 2, 3, 5, 10, 100] {
            let (_, w) = gauss_legendre(-3.0, 7.0, n);
            assert!((w.sum() - 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_polynomial_exactness() {
        // An n-point rule is exact for polynomials up to degree 2n - 1.
        let n = 5;
        let x1: f64 = -1.0;
        let x2: f64 = 2.0;
        for degree in 0..2 * n {
            let degree = i32::try_from(degree).unwrap();
            let expected = (x2.powi(degree + 1) - x1.powi(degree + 1)) / f64::from(degree + 1);
            let actual = integrate(x1, x2, n, |x| x.powi(degree));
            assert!(
                (actual - expected).abs() < 1e-12,
                "degree {degree}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_integrates_exp() {
        let actual = integrate(0.0, 1.0, 8, f64::exp);
        let expected = std::f64::consts::E - 1.0;
        assert!((actual - expected).abs() < 1e-14);
    }

    #[test]
    fn test_high_order_rules() {
        let expected = std::f64::consts::E - 1.0;
        for n in [8, 16, 32, 64, 100] {
            let actual = integrate(0.0, 1.0, n, f64::exp);
            assert!((actual - expected).abs() < 1e-14, "n = {n}: got {actual}");
        }
    }

    #[test]
    fn test_descending_interval() {
        let (x, w) = gauss_legendre(1.0, -1.0, 3);
        // Nodes run from x1 to x2 and weights carry the orientation sign.
        assert!(x[0] > x[1] && x[1] > x[2]);
        assert!(w.iter().all(|&wi| wi < 0.0));
        assert!((w.sum() + 2.0).abs() < 1e-14);
    }
}
