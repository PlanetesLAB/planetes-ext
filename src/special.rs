//! Special functions.

use std::f64::consts::EULER_GAMMA;

/// Exponential integral `Eₙ(x)` for `n >= 1` and `x >= 0`.
///
/// Evaluates the continued fraction for `x > 1` and the power series
/// otherwise, after Press et al. (2007), *Numerical Recipes*, Sect. 6.3.
/// The limiting values `E₁(0) = ∞` and `Eₙ(0) = 1 / (n - 1)` for `n > 1` are
/// returned exactly.
///
/// # Panics
/// Panics if `n == 0`.
///
/// # Examples
/// ```
/// use planetes_ext::special::expint;
///
/// let e1 = expint(1, 1.0);
/// assert!((e1 - 0.2193839343955205).abs() < 1e-14);
///
/// assert_eq!(expint(1, 0.0), f64::INFINITY);
/// assert_eq!(expint(2, 0.0), 1.0);
/// assert_eq!(expint(3, 0.0), 0.5);
/// ```
#[must_use]
#[inline]
pub fn expint(n: u32, x: f64) -> f64 {
    const MAX_ITER: u32 = 200;

    assert!(n >= 1, "n must be at least 1");
    let nm1 = f64::from(n - 1);

    if x == 0.0 {
        return if n == 1 { f64::INFINITY } else { 1.0 / nm1 };
    }

    if x > 1.0 {
        // Continued fraction, evaluated with the modified Lentz method.
        let mut b = x + f64::from(n);
        let mut c = 1.0 / f64::MIN_POSITIVE;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITER {
            let i = f64::from(i);
            let a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() <= f64::EPSILON {
                break;
            }
        }
        h * (-x).exp()
    } else {
        // Power series.
        let mut ans = if n == 1 {
            -x.ln() - EULER_GAMMA
        } else {
            1.0 / nm1
        };
        let mut fact = 1.0;
        for i in 1..=MAX_ITER {
            fact *= -x / f64::from(i);
            let del = if i == n - 1 {
                let psi = -EULER_GAMMA + (1..n).map(|k| 1.0 / f64::from(k)).sum::<f64>();
                fact * (-x.ln() + psi)
            } else {
                -fact / (f64::from(i) - nm1)
            };
            ans += del;
            if del.abs() < ans.abs() * f64::EPSILON {
                break;
            }
        }
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::expint;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_limiting_values_at_zero() {
        assert_eq!(expint(1, 0.0), f64::INFINITY);
        assert_eq!(expint(2, 0.0), 1.0);
        assert_eq!(expint(3, 0.0), 0.5);
        assert_eq!(expint(5, 0.0), 0.25);
    }

    #[test]
    fn test_reference_values() {
        let cases = [
            (1, 0.1, 1.822_923_958_419_390_6),
            (2, 0.1, 0.722_545_022_194_020_4),
            (1, 0.5, 0.559_773_594_776_160_8),
            (2, 0.5, 0.326_643_862_324_553_2),
            (3, 0.5, 0.221_604_364_275_178_46),
            (1, 1.0, 0.219_383_934_395_520_5),
            (2, 1.0, 0.148_495_506_775_921_94),
            (3, 1.0, 0.109_691_967_197_760_2),
            (5, 1.0, 0.070_454_237_461_720_43),
            (1, 2.0, 0.048_900_510_708_061_146),
            (2, 2.0, 0.037_534_261_820_490_47),
            (5, 5.0, 0.000_705_760_693_424_585_3),
            (1, 10.0, 4.156_968_929_685_325e-6),
            (2, 10.0, 3.830_240_465_631_609_5e-6),
        ];

        for (n, x, expected) in cases {
            assert_close(expint(n, x), expected);
        }
    }

    #[test]
    fn test_recurrence() {
        // n E_{n+1}(x) = exp(-x) - x E_n(x)
        for x in [0.1, 0.5, 1.0, 2.0, 10.0] {
            for n in 1..8 {
                let lhs = f64::from(n) * expint(n + 1, x);
                let rhs = (-x).exp() - x * expint(n, x);
                assert_close(lhs, rhs);
            }
        }
    }

    #[test]
    fn test_monotonic_in_x() {
        for n in 1..5 {
            let mut previous = expint(n, 0.0);
            for i in 1..100 {
                let x = f64::from(i) * 0.1;
                let current = expint(n, x);
                assert!(current < previous, "E_{n}({x}) should decrease");
                previous = current;
            }
        }
    }

    #[test]
    #[should_panic(expected = "n must be at least 1")]
    fn test_zero_order_panics() {
        let _ = expint(0, 1.0);
    }
}
