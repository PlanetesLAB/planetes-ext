//! Physics formulas, in CGS units.

use std::f64::consts::PI;

use codata as cc;

const H: f64 = cc::PLANCK_CONSTANT_CGS;
const K: f64 = cc::BOLTZMANN_CONSTANT_CGS;
const C: f64 = cc::SPEED_OF_LIGHT_IN_VACUUM_CGS;
const G: f64 = cc::NEWTONIAN_CONSTANT_OF_GRAVITATION_CGS;

/// A blackbody at a given temperature.
///
/// The spectral radiance is the Planck function, per unit frequency,
/// $$B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{e^{x} - 1}, \quad x = \frac{h \nu}{k T},$$
/// or per unit wavelength,
/// $$B_\lambda(T) = \frac{2 h c^2}{\lambda^5} \frac{1}{e^{x} - 1}, \quad x = \frac{h c}{\lambda k T}.$$
/// All methods are valid for $T \ge 0$ and positive $\nu$ or $\lambda$, and
/// return zero at $T = 0$.
///
/// # Examples
/// ```
/// use planetes_ext::physics::Blackbody;
///
/// let bb = Blackbody::new(100.0);
/// let nu = 1e12;
/// let lam = 2.997_924_58e10 / nu;
///
/// // B_lambda = B_nu c / lambda^2.
/// let ratio = bb.b_lambda(lam) / (bb.b_nu(nu) * 2.997_924_58e10 / (lam * lam));
/// assert!((ratio - 1.0).abs() < 1e-12);
///
/// assert_eq!(Blackbody::new(0.0).b_nu(nu), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blackbody {
    /// $h / (k T)$.
    h_over_kt: f64,
}

impl Blackbody {
    /// A blackbody at temperature `t` (K).
    #[must_use]
    #[inline]
    pub fn new(t: f64) -> Self {
        Self {
            h_over_kt: H / (K * t),
        }
    }

    /// Spectral radiance per unit frequency $B_\nu$ (erg s$^{-1}$ cm$^{-2}$
    /// Hz$^{-1}$ sr$^{-1}$) at frequency `nu` (Hz).
    #[must_use]
    #[inline]
    pub fn b_nu(&self, nu: f64) -> f64 {
        let x = self.h_over_kt * nu;
        2.0 * H / (C * C) * nu * nu * nu / x.exp_m1()
    }

    /// Spectral radiance per unit wavelength $B_\lambda$ (erg s$^{-1}$
    /// cm$^{-2}$ cm$^{-1}$ sr$^{-1}$) at wavelength `lam` (cm).
    #[must_use]
    #[inline]
    pub fn b_lambda(&self, lam: f64) -> f64 {
        let x = self.h_over_kt * C / lam;
        2.0 * H * C * C / lam.powi(5) / x.exp_m1()
    }

    /// Temperature derivative $\partial B_\nu / \partial T$ (erg s$^{-1}$
    /// cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$ K$^{-1}$) at frequency `nu` (Hz),
    /// $$\frac{\partial B_\nu}{\partial T} = \frac{2 k \nu^2}{c^2}
    /// \frac{x^2 e^{x}}{(e^{x} - 1)^2}.$$
    #[must_use]
    #[inline]
    pub fn db_nu_dt(&self, nu: f64) -> f64 {
        let x = self.h_over_kt * nu;
        2.0 * K / (C * C) * nu * nu * x_squared_exp_ratio(x)
    }

    /// Temperature derivative $\partial B_\lambda / \partial T$ (erg s$^{-1}$
    /// cm$^{-2}$ cm$^{-1}$ sr$^{-1}$ K$^{-1}$) at wavelength `lam` (cm),
    /// $$\frac{\partial B_\lambda}{\partial T} = \frac{2 c k}{\lambda^4}
    /// \frac{x^2 e^{x}}{(e^{x} - 1)^2}.$$
    #[must_use]
    #[inline]
    pub fn db_lambda_dt(&self, lam: f64) -> f64 {
        let x = self.h_over_kt * C / lam;
        2.0 * C * K / lam.powi(4) * x_squared_exp_ratio(x)
    }
}

/// $x^2 e^{x} / (e^{x} - 1)^2$, written with $e^{-x}$ so that it does not
/// overflow for large $x$.
#[inline]
fn x_squared_exp_ratio(x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let em1 = (-x).exp_m1();
    x * x * (-x).exp() / (em1 * em1)
}

/// Mean thermal speed $\bar{v} = \sqrt{8 k T / (\pi m)}$ (cm s$^{-1}$) of
/// particles of mass `m` (g) at temperature `t` (K).
#[must_use]
#[inline]
pub fn thermal_speed(t: f64, m: f64) -> f64 {
    (8.0 * K * t / (PI * m)).sqrt()
}

/// Doppler parameter $b = \sqrt{2 k T / m}$ (cm s$^{-1}$), the most probable
/// speed of particles of mass `m` (g) at temperature `t` (K).
#[must_use]
#[inline]
pub fn doppler_b(t: f64, m: f64) -> f64 {
    (2.0 * K * t / m).sqrt()
}

/// Isothermal sound speed $c_{\mathrm{s}} = \sqrt{k T / m}$ (cm s$^{-1}$) of a gas with
/// mean particle mass `m` (g) at temperature `t` (K).
#[must_use]
#[inline]
pub fn sound_speed(t: f64, m: f64) -> f64 {
    (K * t / m).sqrt()
}

/// Keplerian orbital speed $v_{\mathrm{K}} = \sqrt{G M / r}$ (cm s$^{-1}$) at radius `r`
/// (cm) around a mass `m` (g).
#[must_use]
#[inline]
pub fn kepler_velocity(m: f64, r: f64) -> f64 {
    (G * m / r).sqrt()
}

/// Keplerian angular frequency $\Omega_{\mathrm{K}} = \sqrt{G M / r^3}$ (s$^{-1}$) at
/// radius `r` (cm) around a mass `m` (g).
#[must_use]
#[inline]
pub fn omega_kepler(m: f64, r: f64) -> f64 {
    (G * m / (r * r * r)).sqrt()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::quadrature::gauss_legendre;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn b_nu_matches_direct_formula() {
        for &t in &[10.0, 100.0, 1e4] {
            let bb = Blackbody::new(t);
            for &nu in &[1e9, 1e11, 1e13, 1e15] {
                let x = H * nu / (K * t);
                // Below x ~ 1e-2 the reference itself loses digits to cancellation.
                if !(1e-2..700.0).contains(&x) {
                    continue;
                }
                let direct = 2.0 * H * nu.powi(3) / (C * C) / (x.exp() - 1.0);
                assert!(rel(bb.b_nu(nu), direct) < 1e-12, "t={t} nu={nu}");
            }
        }
    }

    #[test]
    fn rayleigh_jeans_and_wien_limits() {
        let t = 1000.0;
        let bb = Blackbody::new(t);
        let nu_rj = 1e-8 * K * t / H;
        assert!(rel(bb.b_nu(nu_rj), 2.0 * nu_rj * nu_rj * K * t / (C * C)) < 1e-8);
        let nu_w = 300.0 * K * t / H;
        let wien = 2.0 * H * nu_w.powi(3) / (C * C) * (-H * nu_w / (K * t)).exp();
        assert!(rel(bb.b_nu(nu_w), wien) < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_temperature_and_overflow_give_zero() {
        let cold = Blackbody::new(0.0);
        assert_eq!(cold.b_nu(1e12), 0.0);
        assert_eq!(cold.b_lambda(1e-4), 0.0);
        assert_eq!(cold.db_nu_dt(1e12), 0.0);
        assert_eq!(cold.db_lambda_dt(1e-4), 0.0);
        let bb = Blackbody::new(1.0);
        assert_eq!(bb.b_nu(1e15), 0.0);
        assert_eq!(bb.db_nu_dt(1e15), 0.0);
    }

    #[test]
    fn b_lambda_matches_b_nu() {
        let bb = Blackbody::new(30.0);
        for &lam in &[3e-4, 1e-3, 1e-2, 1.0] {
            let nu = C / lam;
            assert!(rel(bb.b_lambda(lam), bb.b_nu(nu) * C / (lam * lam)) < 1e-12);
            assert!(rel(bb.db_lambda_dt(lam), bb.db_nu_dt(nu) * C / (lam * lam)) < 1e-12);
        }
    }

    #[test]
    fn derivative_matches_finite_difference() {
        let t = 50.0;
        let dt = 1e-4 * t;
        for &nu in &[1e10, 1e12, 1e13] {
            let fd = (Blackbody::new(t + dt).b_nu(nu) - Blackbody::new(t - dt).b_nu(nu))
                / (2.0 * dt);
            assert!(rel(Blackbody::new(t).db_nu_dt(nu), fd) < 1e-7, "nu={nu}");
        }
    }

    #[test]
    fn frequency_integral_is_stefan_boltzmann() {
        let t = 300.0;
        let bb = Blackbody::new(t);
        let (x, w) = gauss_legendre(1e-10, 80.0, 400);
        let nu_per_x = K * t / H;
        let integral: f64 = x
            .iter()
            .zip(w.iter())
            .map(|(&xi, &wi)| wi * bb.b_nu(xi * nu_per_x) * nu_per_x)
            .sum();
        let sigma = cc::STEFAN_BOLTZMANN_CONSTANT_CGS;
        assert!(rel(PI * integral, sigma * t.powi(4)) < 1e-10);
    }

    #[test]
    fn thermal_speeds() {
        let (t, m) = (1e4, cc::PROTON_MASS_CGS);
        assert!(rel(doppler_b(t, m), 1.2849e6) < 1e-4);
        assert!(rel(thermal_speed(t, m), 2.0 / PI.sqrt() * doppler_b(t, m)) < 1e-14);
        assert!(rel(sound_speed(t, m), doppler_b(t, m) / 2f64.sqrt()) < 1e-14);
    }

    #[test]
    fn earth_orbit() {
        let (m, r) = (cc::astro::M_SUN_CGS, cc::astro::AU_CGS);
        assert!(rel(kepler_velocity(m, r), 2.978e6) < 1e-3);
        assert!(rel(omega_kepler(m, r) * r, kepler_velocity(m, r)) < 1e-14);
        let period_days = 2.0 * PI / omega_kepler(m, r) / 86_400.0;
        assert!(rel(period_days, 365.2569) < 1e-4);
    }
}
