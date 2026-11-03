use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1};
use num_traits::Zero;

/// Generic Vector (1D array)
pub type Vector<T> = Array1<T>;

/// n-dimensional real vector (1D array).
pub type RVector = Array1<f64>;

/// n-dimensional real vector view (1D view).
pub type RVecView<'a> = ArrayView1<'a, f64>;

/// n-dimensional real vector mutable view (1D mutable view).
pub type RVecViewMut<'a> = ndarray::ArrayViewMut1<'a, f64>;

/// Generic matrix (2D array)
pub type Matrix<T> = Array2<T>;

/// 2-dimensional unsigned-integer matrix.
pub type UMatrix = Array2<usize>;

/// A real matrix (2D ndarray).
pub type RMatrix = Array2<f64>;

/// n-dimensional real matrix view (2D view)
pub type RMatView<'a> = ndarray::ArrayView2<'a, f64>;

/// n-dimensional real matrix mutable view (2D mutable view)
pub type RMatViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

/// Generic tensor (3D array)
pub type Tensor<T> = Array3<T>;

/// A real tensor (3D ndarray).
pub type RTensor = Array3<f64>;

/// A 4-dimensional real tensor (4D ndarray).
pub type RTensor4 = Array4<f64>;

/// 1-dimensional unsigned-integer vector.
pub type UVector = Array1<usize>;

/// 1-dimensional signed-integer vector.
pub type IVector = Array1<isize>;

/// 1-dimensional boolean vector.
pub type BVector = Array1<bool>;

#[cfg(feature = "complex")]
use num_complex::Complex;

#[cfg(feature = "complex")]
/// A fixed-length array of complex ([f64]) numbers.
pub type CVector = Array1<Complex<f64>>;

#[cfg(feature = "complex")]
/// A 2-dimensional array (matrix) of complex ([f64]) numbers.
pub type CMatrix = Array2<Complex<f64>>;

#[cfg(feature = "complex")]
/// A 3-dimensional array (tensor) of complex ([f64]) numbers.
pub type CTensor = Array3<Complex<f64>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[must_use]
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    #[inline]
    pub fn to_array(&self) -> RVector {
        ndarray::array![self.x, self.y, self.z]
    }

    /// Creates a `Vec3` from a 1D array. The array must have exactly 3 elements.
    ///
    /// # Panics
    /// Panics if the input array does not have exactly 3 elements.
    #[must_use]
    #[inline]
    pub fn from_array(arr: &RVector) -> Self {
        assert_eq!(arr.len(), 3, "Array must have exactly 3 elements");
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }

    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    #[must_use]
    #[inline]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[must_use]
    #[inline]
    pub fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    #[must_use]
    #[inline]
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[must_use]
    #[inline]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[must_use]
    #[inline]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

// Vector * scalar
impl Mul<f64> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

// scalar * Vector (optional but often useful)
impl Mul<Vec3> for f64 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

// Vector + Vector
impl Add for Vec3 {
    type Output = Vec3;

    #[inline]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    #[inline]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn div(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

// Enable iterator .sum()
impl Sum for Vec3 {
    #[inline]
    fn sum<I: Iterator<Item = Vec3>>(iter: I) -> Vec3 {
        iter.fold(Vec3::zero(), |a, b| a + b)
    }
}

impl Zero for Vec3 {
    #[inline]
    fn zero() -> Self {
        Vec3::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    #[inline]
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
