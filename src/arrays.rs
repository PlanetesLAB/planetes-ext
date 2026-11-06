use std::fmt;

use ndarray::{ArrayBase, Axis, Ix1, OwnedRepr};
use ndarray::{Data, Dimension, RemoveAxis};

use crate::types::RVector;

/// Error type for extrema operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremaError {
    EmptyArray,
    UndefinedOrder, // e.g., NaN encountered
}

impl fmt::Display for ExtremaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtremaError::EmptyArray => write!(f, "cannot compute extrema of empty array"),
            ExtremaError::UndefinedOrder => {
                write!(f, "undefined order: encountered NaN or incomparable values")
            }
        }
    }
}

impl std::error::Error for ExtremaError {}

/// Extension trait providing convenience methods for computing extrema
/// (minima, maxima) and their indices on [`ndarray::ArrayBase`] values.
pub trait ArrayExtrema<T, D>
where
    D: Dimension,
{
    /// Returns the maximum value in the array.
    ///
    /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use planetes_ext::arrays::ArrayExtrema;
    /// use planetes_ext::arrays::ExtremaError;
    ///
    /// let a = array![1, 3, 2];
    /// assert_eq!(a.maxval(), Ok(3));
    ///
    /// let empty: ndarray::Array1<i32> = ndarray::Array1::from_vec(vec![]);
    /// assert_eq!(empty.maxval(), Err(ExtremaError::EmptyArray));
    /// ```
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn maxval(&self) -> Result<T, ExtremaError>;

    /// Returns the minimum value in the array.
    ///
    /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn minval(&self) -> Result<T, ExtremaError>;

    /// Returns the minimum and maximum values in the array.
    ///
    /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn minmax(&self) -> Result<(T, T), ExtremaError>;

    /// Returns an array of maximum values along the given axis.
    ///
    /// Each element of the returned array is the maximum of the slice taken
    /// along axis. Returns `Err(ExtremaError::UndefinedOrder)` if NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Panics
    /// Panics if any subview is empty, though this cannot occur if self
    /// itself is non-empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn maxval_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError>;

    /// Returns an array of minimum values along the given axis.
    ///
    /// Each element of the returned array is the minimum of the slice taken
    /// along axis. Returns `Err(ExtremaError::UndefinedOrder)` if NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn minval_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError>;

    /// Returns the index of the maximum element in the array.
    ///
    /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// The index is returned in [`ndarray::Dimension::Pattern`] form, which matches the array's dimensionality.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn argmax(&self) -> Result<D::Pattern, ExtremaError>;

    /// Returns the index of the minimum element in the array.
    ///
    /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
    /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn argmin(&self) -> Result<D::Pattern, ExtremaError>;

    /// Returns an array of indices of the maximum elements along the given axis.
    ///
    /// Each element in the returned array is the index (within the axis) of the
    /// maximum value of the corresponding subview. Returns `Err(ExtremaError::UndefinedOrder)`
    /// if NaN values are encountered, or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn argmax_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError>;

    /// Returns an array of indices of the minimum elements along the given axis.
    ///
    /// Each element in the returned array is the index (within the axis) of the
    /// minimum value of the corresponding subview. Returns `Err(ExtremaError::UndefinedOrder)`
    /// if NaN values are encountered, or `Err(ExtremaError::EmptyArray)` if the array is empty.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    /// * `Err(ExtremaError::UndefinedOrder)` if any element is NaN or otherwise incomparable.
    fn argmin_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError>;
}

impl<T, S, D> ArrayExtrema<T, D> for ArrayBase<S, D>
where
    T: PartialOrd + Copy,
    S: Data<Elem = T>,
    D: Dimension + RemoveAxis,
{
    #[inline]
    fn maxval(&self) -> Result<T, ExtremaError> {
        let mut iter = self.iter();
        let &first = iter.next().ok_or(ExtremaError::EmptyArray)?;
        if first.partial_cmp(&first).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }
        let mut max_val = first;
        for &val in iter {
            match val.partial_cmp(&max_val) {
                Some(std::cmp::Ordering::Greater) => max_val = val,
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
        }
        Ok(max_val)
    }

    #[inline]
    fn minval(&self) -> Result<T, ExtremaError> {
        let mut iter = self.iter();
        let &first = iter.next().ok_or(ExtremaError::EmptyArray)?;
        if first.partial_cmp(&first).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }
        let mut min_val = first;
        for &val in iter {
            match val.partial_cmp(&min_val) {
                Some(std::cmp::Ordering::Less) => min_val = val,
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
        }
        Ok(min_val)
    }

    #[inline]
    fn minmax(&self) -> Result<(T, T), ExtremaError> {
        let mut iter = self.iter();
        let &first = iter.next().ok_or(ExtremaError::EmptyArray)?;
        if first.partial_cmp(&first).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }
        let mut min_val = first;
        let mut max_val = first;
        for &val in iter {
            match val.partial_cmp(&min_val) {
                Some(std::cmp::Ordering::Less) => min_val = val,
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
            match val.partial_cmp(&max_val) {
                Some(std::cmp::Ordering::Greater) => max_val = val,
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
        }
        Ok((min_val, max_val))
    }

    #[inline]
    fn maxval_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let first = self.index_axis(axis, 0);
        for &val in &first {
            if val.partial_cmp(&val).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }
        }
        let mut result = first.to_owned();

        for i in 1..self.len_of(axis) {
            let lane = self.index_axis(axis, i);
            for (r, &v) in result.iter_mut().zip(lane.iter()) {
                match v.partial_cmp(r) {
                    Some(std::cmp::Ordering::Greater) => *r = v,
                    Some(_) => {}
                    None => return Err(ExtremaError::UndefinedOrder),
                }
            }
        }

        Ok(result)
    }

    #[inline]
    fn minval_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let first = self.index_axis(axis, 0);
        for &val in &first {
            if val.partial_cmp(&val).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }
        }
        let mut result = first.to_owned();

        for i in 1..self.len_of(axis) {
            let lane = self.index_axis(axis, i);
            for (r, &v) in result.iter_mut().zip(lane.iter()) {
                match v.partial_cmp(r) {
                    Some(std::cmp::Ordering::Less) => *r = v,
                    Some(_) => {}
                    None => return Err(ExtremaError::UndefinedOrder),
                }
            }
        }

        Ok(result)
    }

    #[inline]
    fn argmax(&self) -> Result<D::Pattern, ExtremaError> {
        let mut iter = self.indexed_iter();
        let (first_idx, &first_val) = iter.next().ok_or(ExtremaError::EmptyArray)?;
        if first_val.partial_cmp(&first_val).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }
        let mut best_idx = first_idx;
        let mut best_val = first_val;
        for (idx, &val) in iter {
            match val.partial_cmp(&best_val) {
                Some(std::cmp::Ordering::Greater) => {
                    best_idx = idx;
                    best_val = val;
                }
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
        }
        Ok(best_idx)
    }

    #[inline]
    fn argmin(&self) -> Result<D::Pattern, ExtremaError> {
        let mut iter = self.indexed_iter();
        let (first_idx, &first_val) = iter.next().ok_or(ExtremaError::EmptyArray)?;
        if first_val.partial_cmp(&first_val).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }
        let mut best_idx = first_idx;
        let mut best_val = first_val;
        for (idx, &val) in iter {
            match val.partial_cmp(&best_val) {
                Some(std::cmp::Ordering::Less) => {
                    best_idx = idx;
                    best_val = val;
                }
                Some(_) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }
        }
        Ok(best_idx)
    }

    #[inline]
    fn argmax_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let first = self.index_axis(axis, 0);
        for &val in &first {
            if val.partial_cmp(&val).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }
        }
        let mut best_vals = first.to_owned();
        let mut best_idxs = first.map(|_| 0usize);

        for i in 1..self.len_of(axis) {
            let lane = self.index_axis(axis, i);
            for ((bv, bi), &v) in best_vals
                .iter_mut()
                .zip(best_idxs.iter_mut())
                .zip(lane.iter())
            {
                match v.partial_cmp(bv) {
                    Some(std::cmp::Ordering::Greater) => {
                        *bv = v;
                        *bi = i;
                    }
                    Some(_) => {}
                    None => return Err(ExtremaError::UndefinedOrder),
                }
            }
        }

        Ok(best_idxs)
    }

    #[inline]
    fn argmin_along(
        &self,
        axis: Axis,
    ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let first = self.index_axis(axis, 0);
        for &val in &first {
            if val.partial_cmp(&val).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }
        }
        let mut best_vals = first.to_owned();
        let mut best_idxs = first.map(|_| 0usize);

        for i in 1..self.len_of(axis) {
            let lane = self.index_axis(axis, i);
            for ((bv, bi), &v) in best_vals
                .iter_mut()
                .zip(best_idxs.iter_mut())
                .zip(lane.iter())
            {
                match v.partial_cmp(bv) {
                    Some(std::cmp::Ordering::Less) => {
                        *bv = v;
                        *bi = i;
                    }
                    Some(_) => {}
                    None => return Err(ExtremaError::UndefinedOrder),
                }
            }
        }

        Ok(best_idxs)
    }
}

/// Trait providing monotonicity checks for 1-D ndarray arrays.
pub trait Sequence<T>
where
    T: PartialOrd + Copy,
{
    /// Returns true if the array is monotonically (non-strictly) increasing.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_monotonically_increasing(&self) -> Result<bool, ExtremaError>;

    /// Returns true if the array is monotonically (non-strictly) decreasing.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_monotonically_decreasing(&self) -> Result<bool, ExtremaError>;

    /// Returns true if the array is strictly increasing.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_strictly_increasing(&self) -> Result<bool, ExtremaError>;

    /// Returns true if the array is strictly decreasing.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_strictly_decreasing(&self) -> Result<bool, ExtremaError>;

    /// Returns true if the array is strictly monotonic.
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_strictly_monotonic(&self) -> Result<bool, ExtremaError>;

    /// Returns true if the array is monotonic (non-decreasing or non-increasing).
    ///
    /// # Errors
    /// * `Err(ExtremaError::EmptyArray)` if the array has no elements.
    fn is_monotonic(&self) -> Result<bool, ExtremaError>;
}

impl<T, S> Sequence<T> for ArrayBase<S, Ix1>
where
    T: PartialOrd + Copy,
    S: Data<Elem = T>,
{
    #[inline]
    fn is_monotonically_increasing(&self) -> Result<bool, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(std::cmp::Ordering::Less) => return Ok(false),
                Some(_) => {} // >= ok
                None => return Err(ExtremaError::UndefinedOrder),
            }

            prev = curr;
        }

        Ok(true)
    }

    #[inline]
    fn is_monotonically_decreasing(&self) -> Result<bool, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(std::cmp::Ordering::Greater) => return Ok(false),
                Some(_) => {} // <= ok
                None => return Err(ExtremaError::UndefinedOrder),
            }

            prev = curr;
        }

        Ok(true)
    }

    #[inline]
    fn is_strictly_increasing(&self) -> Result<bool, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(std::cmp::Ordering::Greater) => {} // required
                _ => return Ok(false),                  // <= means not strictly
            }

            prev = curr;
        }

        Ok(true)
    }

    #[inline]
    fn is_strictly_decreasing(&self) -> Result<bool, ExtremaError> {
        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(std::cmp::Ordering::Less) => {} // required
                _ => return Ok(false),               // >= means not strictly
            }

            prev = curr;
        }

        Ok(true)
    }

    #[inline]
    fn is_strictly_monotonic(&self) -> Result<bool, ExtremaError> {
        use std::cmp::Ordering;

        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        let mut direction: Option<Ordering> = None;

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(Ordering::Greater) => {
                    if direction == Some(Ordering::Less) {
                        return Ok(false);
                    }
                    direction = Some(Ordering::Greater);
                }
                Some(Ordering::Less) => {
                    if direction == Some(Ordering::Greater) {
                        return Ok(false);
                    }
                    direction = Some(Ordering::Less);
                }
                Some(Ordering::Equal) => return Ok(false),
                None => return Err(ExtremaError::UndefinedOrder),
            }

            prev = curr;
        }

        Ok(true)
    }

    #[inline]
    fn is_monotonic(&self) -> Result<bool, ExtremaError> {
        use std::cmp::Ordering;

        if self.is_empty() {
            return Err(ExtremaError::EmptyArray);
        }

        let mut iter = self.iter().copied();
        let mut prev = iter.next().unwrap();

        if prev.partial_cmp(&prev).is_none() {
            return Err(ExtremaError::UndefinedOrder);
        }

        let mut seen_increase = false;
        let mut seen_decrease = false;

        for curr in iter {
            if curr.partial_cmp(&curr).is_none() {
                return Err(ExtremaError::UndefinedOrder);
            }

            match curr.partial_cmp(&prev) {
                Some(Ordering::Greater) => seen_increase = true,
                Some(Ordering::Less) => seen_decrease = true,
                Some(Ordering::Equal) => {}
                None => return Err(ExtremaError::UndefinedOrder),
            }

            if seen_increase && seen_decrease {
                return Ok(false);
            }

            prev = curr;
        }

        Ok(true)
    }
}

pub trait Integrable {
    fn trapezoid(&self, x: &RVector) -> f64;

    /// Computes the integral using Simpson's rule.
    ///
    /// `x` must be sorted in strictly ascending order; spacing may be
    /// non-uniform. When the number of points is even, Simpson's 3/8 rule is
    /// applied to the last three intervals.
    ///
    /// # Panics
    /// Panics if `self` and `x` have different lengths.
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use planetes_ext::arrays::Integrable;
    ///
    /// let x = array![0.0, 1.0, 2.0];
    /// let y = array![0.0, 1.0, 8.0]; // y = x^3
    /// assert_eq!(y.simpson(&x), 4.0);
    /// ```
    fn simpson(&self, x: &RVector) -> f64;
}

/// Simpson's 1/3 rule over one panel of three points.
#[inline]
fn simpson_panel(x: &[f64], y: &[f64]) -> f64 {
    let h0 = x[1] - x[0];
    let h1 = x[2] - x[1];
    let hsum = h0 + h1;
    hsum / 6.0
        * (y[0] * (2.0 - h1 / h0) + y[1] * (hsum * hsum / (h0 * h1)) + y[2] * (2.0 - h0 / h1))
}

/// Simpson's 3/8 rule over the last three intervals of four points.
#[inline]
fn simpson_38_tail(x: &[f64], y: &[f64]) -> f64 {
    let h0 = x[1] - x[0];
    let h1 = x[2] - x[1];
    let h2 = x[3] - x[2];
    let d = h0 + h1 + h2;
    let d3 = d * d * d;
    let w0 = d * (3.0 * h0 * h0 + 2.0 * h0 * h1 - 2.0 * h0 * h2 - h1 * h1 + h2 * h2)
        / (12.0 * h0 * (h0 + h1));
    let w1 = (h0 + h1 - h2) * d3 / (12.0 * h0 * h1 * (h1 + h2));
    let w2 = (h1 + h2 - h0) * d3 / (12.0 * h1 * h2 * (h0 + h1));
    let w3 = d * (h0 * h0 - 2.0 * h0 * h2 - h1 * h1 + 2.0 * h1 * h2 + 3.0 * h2 * h2)
        / (12.0 * h2 * (h1 + h2));
    w0 * y[0] + w1 * y[1] + w2 * y[2] + w3 * y[3]
}

impl Integrable for RVector {
    #[inline]
    fn trapezoid(&self, x: &RVector) -> f64 {
        assert_eq!(self.len(), x.len(), "Arrays must have the same length");
        let n = self.len();
        if n < 2 {
            return 0.0;
        }
        let ys = self.as_slice().expect("y array must be contiguous");
        let xs = x.as_slice().expect("x array must be contiguous");
        let mut sum = 0.0;
        for i in 0..n - 1 {
            sum += (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]);
        }
        sum * 0.5
    }

    #[inline]
    fn simpson(&self, x: &RVector) -> f64 {
        assert_eq!(self.len(), x.len(), "Arrays must have the same length");
        let n = self.len();
        if n < 2 {
            return 0.0;
        }
        let ys = self.as_slice().expect("y array must be contiguous");
        let xs = x.as_slice().expect("x array must be contiguous");

        if n == 2 {
            return (ys[0] + ys[1]) * (xs[1] - xs[0]) * 0.5;
        }

        let panels = xs.windows(3).step_by(2).zip(ys.windows(3).step_by(2));
        let mut sum = 0.0;
        if n % 2 == 1 {
            for (xp, yp) in panels {
                sum += simpson_panel(xp, yp);
            }
        } else {
            // Leave the last three intervals for Simpson's 3/8 rule.
            for (xp, yp) in panels.take((n - 4) / 2) {
                sum += simpson_panel(xp, yp);
            }
            sum += simpson_38_tail(&xs[n - 4..], &ys[n - 4..]);
        }
        sum
    }
}

pub trait AsRealSlice {
    /// Returns a reference to the underlying real slice, assuming it is contiguous (standard layout).
    fn as_real_slice(&self) -> &[f64];
}

impl<S> AsRealSlice for ArrayBase<S, Ix1>
where
    S: Data<Elem = f64>,
{
    #[inline]
    fn as_real_slice(&self) -> &[f64] {
        self.as_slice().expect("1-D Array assumed contiguous.")
    }
}

/// Returns the index of the largest element in a sorted slice that is
/// less than or equal to the given value.
///
/// # Arguments
/// * `val` - The target value to compare against.
/// * `array` - A slice of `f64` values. Must be sorted in non-decreasing order.
///
/// # Returns
/// * `Some(index)` if an element exists in `array` such that:
///   - `array[index] <= val`
///   - and `array[index + 1] > val` (or `index` is the last valid element).
/// * `None` if:
///   - the slice is empty
///   - `val` is smaller than the first element
///   - `val` is greater than or equal to the last element
///
/// # Complexity
/// Runs in O(log n) time using binary search via `partition_point`.
///
/// # Example
/// ```
/// use planetes_ext::arrays::find_index_le;
///
/// let arr = [1.0, 2.5, 4.0, 7.0];
/// assert_eq!(find_index_le(3.0, &arr), Some(1)); // arr[1] = 2.5
/// assert_eq!(find_index_le(1.0, &arr), Some(0));
/// assert_eq!(find_index_le(7.0, &arr), None);
/// assert_eq!(find_index_le(0.5, &arr), None);
/// ```
#[must_use]
#[inline]
pub fn find_index_le(val: f64, array: &[f64]) -> Option<usize> {
    if array.is_empty() || val < array[0] || val >= array[array.len() - 1] {
        return None;
    }
    let idx = array.partition_point(|&x| x <= val);
    if idx > 0 { Some(idx - 1) } else { None }
}

/// Returns the index of the smallest element in a sorted slice that is
/// greater than or equal to the given value.
///
/// # Arguments
/// * `val` - The target value to compare against.
/// * `array` - A slice of `f64` values. Must be sorted in non-decreasing order.
///
/// # Returns
/// * `Some(index)` if an element exists in `array` such that:
///   - `array[index] >= val`
///   - and `array[index - 1] < val` (or `index` is the first valid element).
/// * `None` if:
///   - the slice is empty
///   - `val` is smaller than or equal to the first element
///   - `val` is greater than the last element
///
/// # Complexity
/// Runs in O(log n) time using binary search via `partition_point`.
///
/// # Example
/// ```
/// use planetes_ext::arrays::find_index_ge;
///
/// let arr = [1.0, 2.5, 4.0, 7.0];
/// assert_eq!(find_index_ge(3.0, &arr), Some(2)); // arr[2] = 4.0
/// assert_eq!(find_index_ge(2.5, &arr), Some(1));
/// assert_eq!(find_index_ge(0.5, &arr), None);
/// assert_eq!(find_index_ge(8.0, &arr), None);
/// ```
#[must_use]
#[inline]
pub fn find_index_ge(val: f64, array: &[f64]) -> Option<usize> {
    if array.is_empty() || val > array[array.len() - 1] || val <= array[0] {
        return None;
    }
    let idx = array.partition_point(|&x| x < val);
    if idx < array.len() { Some(idx) } else { None }
}

/// Returns the index of the first element in a sorted slice that is
/// **greater than or equal to** `val`.
///
/// This is equivalent to [`find_index_ge`] but clamps the result to valid indices.
/// If `val` is less than or equal to the first element, returns `0`.
/// If `val` is greater than or equal to the last element, returns `array.len()`.
///
/// # Examples
/// ```
/// use planetes_ext::arrays::lower_bound_index;
///
/// let arr = [1.0, 3.0, 5.0, 7.0];
///
/// // Insert before any elements ≥ 4.0 → index 2
/// assert_eq!(lower_bound_index(4.0, &arr), 2);
///
/// // Value below first → index 0
/// assert_eq!(lower_bound_index(0.5, &arr), 0);
///
/// // Value above last → index len = 4
/// assert_eq!(lower_bound_index(10.0, &arr), arr.len());
/// ```
/// # See Also
/// [`upper_bound_index`], [`find_index_ge`]
#[must_use]
#[inline]
pub fn lower_bound_index(val: f64, array: &[f64]) -> usize {
    if array.is_empty() {
        return 0;
    }

    if val <= array[0] {
        return 0;
    }

    if val >= array[array.len() - 1] {
        return array.len();
    }

    array.partition_point(|&x| x < val)
}

/// Returns the index of the **last element ≤ `val`** in a sorted slice.
///
/// This behaves similarly to [`find_index_le`] but clamps the result to valid indices.
/// If `val` is less than the first element, returns `0`.
/// If `val` is greater than or equal to the last element, returns `array.len() - 1`.
///
/// # Examples
/// ```
/// use planetes_ext::arrays::upper_bound_index;
/// let arr = [1.0, 3.0, 5.0, 7.0];
///
/// // Largest element ≤ 4.0 is 3.0 at index 1
/// assert_eq!(upper_bound_index(4.0, &arr), 1);
///
/// // Value below first → clamp to 0
/// assert_eq!(upper_bound_index(0.5, &arr), 0);
///
/// // Value above last → clamp to last index = 3
/// assert_eq!(upper_bound_index(10.0, &arr), arr.len() - 1);
/// ```
/// # See Also
/// [`lower_bound_index`], [`find_index_le`]
#[must_use]
#[inline]
pub fn upper_bound_index(val: f64, array: &[f64]) -> usize {
    if array.is_empty() {
        return 0;
    }

    if val < array[0] {
        return 0;
    }

    if val >= array[array.len() - 1] {
        return array.len() - 1;
    }

    array.partition_point(|&x| x <= val) - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn test_maxval_minval_nonempty() {
        let a = array![1.0, 3.5, 2.2, -5.1, 7.3];
        assert_eq!(a.maxval(), Ok(7.3));
        assert_eq!(a.minval(), Ok(-5.1));
    }

    #[test]
    fn test_maxval_minval_empty() {
        let a: Array1<f64> = Array1::from_vec(vec![]);
        assert_eq!(a.maxval(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.minval(), Err(ExtremaError::EmptyArray));
    }

    #[test]
    fn test_maxval_along_axis0() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.maxval_along(Axis(0)).unwrap();
        assert_eq!(result, array![3.3, 4.2, 7.7]);
    }

    #[test]
    fn test_minval_along_axis0() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.minval_along(Axis(0)).unwrap();
        assert_eq!(result, array![1.0, -1.5, 2.1]);
    }

    #[test]
    fn test_maxval_along_axis1() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.maxval_along(Axis(1)).unwrap();
        assert_eq!(result, array![4.2, 7.7]);
    }

    #[test]
    fn test_minval_along_axis1() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.minval_along(Axis(1)).unwrap();
        assert_eq!(result, array![1.0, -1.5]);
    }

    #[test]
    fn test_argmax_argmin() {
        let a = array![10.0, 3.1, 50.5, -2.2, 50.5];
        assert_eq!(a.argmax(), Ok(2)); // first 50.5
        assert_eq!(a.argmin(), Ok(3));
    }

    #[test]
    fn test_argmax_along_axis0() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.argmax_along(Axis(0)).unwrap();
        assert_eq!(result, array![1, 0, 1]);
    }

    #[test]
    fn test_argmin_along_axis0() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.argmin_along(Axis(0)).unwrap();
        assert_eq!(result, array![0, 1, 0]);
    }

    #[test]
    fn test_argmax_along_axis1() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.argmax_along(Axis(1)).unwrap();
        assert_eq!(result, array![1, 2]);
    }

    #[test]
    fn test_argmin_along_axis1() {
        let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
        let result = a.argmin_along(Axis(1)).unwrap();
        assert_eq!(result, array![0, 1]);
    }

    #[test]
    fn test_maxval_minval_with_nan() {
        let a = array![1.0, f64::NAN, 3.5];
        // Now should return UndefinedOrder error when NaN is present
        assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_maxval_along_axis_with_nan() {
        let a = array![[1.0, f64::NAN, 2.0], [3.0, 4.0, 5.0]];

        // Along axis 0: column 1 has a NaN, so should return UndefinedOrder error
        assert_eq!(a.maxval_along(Axis(0)), Err(ExtremaError::UndefinedOrder));

        // Along axis 1: first row has a NaN, so should return UndefinedOrder error
        assert_eq!(a.maxval_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_minval_along_axis_with_nan() {
        let a = array![[1.0, 4.0, 2.0], [f64::NAN, -1.5, 7.0]];

        // Both axes should return UndefinedOrder error due to NaN presence
        assert_eq!(a.minval_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.minval_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_argmax_argmin_with_nan() {
        let a = array![1.0, f64::NAN, 3.5];
        // Should return UndefinedOrder error when NaN is present
        assert_eq!(a.argmax(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmin(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_argmax_argmin_along_with_nan() {
        let a = array![[1.0, f64::NAN, 2.0], [3.0, 4.0, 5.0]];

        // Should return UndefinedOrder error due to NaN presence
        assert_eq!(a.argmax_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmin_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmax_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmin_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_all_methods_empty_2d() {
        let a: Array2<f64> = Array2::from_shape_vec((0, 3), vec![]).unwrap();
        assert_eq!(a.maxval(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.minval(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.maxval_along(Axis(0)), Err(ExtremaError::EmptyArray));
        assert_eq!(a.minval_along(Axis(1)), Err(ExtremaError::EmptyArray));
        assert_eq!(a.argmax(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.argmin(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.argmax_along(Axis(0)), Err(ExtremaError::EmptyArray));
        assert_eq!(a.argmin_along(Axis(1)), Err(ExtremaError::EmptyArray));
    }

    #[test]
    fn test_valid_arrays_without_nan() {
        // Test that normal arrays (without NaN) work correctly
        let a = array![1, 5, 3, 2, 4];
        assert_eq!(a.maxval(), Ok(5));
        assert_eq!(a.minval(), Ok(1));
        assert_eq!(a.argmax(), Ok(1));
        assert_eq!(a.argmin(), Ok(0));

        let b = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(b.maxval_along(Axis(0)).unwrap(), array![4, 5, 6]);
        assert_eq!(b.minval_along(Axis(0)).unwrap(), array![1, 2, 3]);
        assert_eq!(b.argmax_along(Axis(1)).unwrap(), array![2, 2]);
        assert_eq!(b.argmin_along(Axis(1)).unwrap(), array![0, 0]);
    }

    #[test]
    fn test_single_element_arrays() {
        let a = array![42.0];
        assert_eq!(a.maxval(), Ok(42.0));
        assert_eq!(a.minval(), Ok(42.0));
        assert_eq!(a.argmax(), Ok(0));
        assert_eq!(a.argmin(), Ok(0));

        let b = array![[5.0]];
        assert_eq!(b.maxval_along(Axis(0)).unwrap(), array![5.0]);
        assert_eq!(b.minval_along(Axis(1)).unwrap(), array![5.0]);
        assert_eq!(b.argmax_along(Axis(0)).unwrap(), array![0]);
        assert_eq!(b.argmin_along(Axis(1)).unwrap(), array![0]);
    }

    #[test]
    fn test_single_nan_element() {
        let a = array![f64::NAN];
        assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmax(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.argmin(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_increasing_true() {
        let a = array![1.0, 2.0, 2.0, 5.0];
        assert!(a.is_monotonically_increasing().unwrap());
        assert!(!a.is_strictly_increasing().unwrap());
    }

    #[test]
    fn test_increasing_false() {
        let a = array![1.0, 3.0, 2.0];
        assert!(!a.is_monotonically_increasing().unwrap());
        assert!(!a.is_strictly_increasing().unwrap());
    }

    #[test]
    fn test_strictly_increasing_true() {
        let a = array![1.0, 2.0, 3.0];
        assert!(a.is_strictly_increasing().unwrap());
        assert!(a.is_monotonically_increasing().unwrap());
    }

    #[test]
    fn test_decreasing_true() {
        let a = array![5.0, 4.0, 4.0, 1.0];
        assert!(a.is_monotonically_decreasing().unwrap());
        assert!(!a.is_strictly_decreasing().unwrap());
    }

    #[test]
    fn test_decreasing_false() {
        let a = array![5.0, 3.0, 4.0];
        assert!(!a.is_monotonically_decreasing().unwrap());
        assert!(!a.is_strictly_decreasing().unwrap());
    }

    #[test]
    fn test_strictly_decreasing_true() {
        let a = array![5.0, 3.0, 1.0];
        assert!(a.is_strictly_decreasing().unwrap());
        assert!(a.is_monotonically_decreasing().unwrap());
    }

    #[test]
    fn test_empty_array() {
        let a = Array1::<f64>::zeros(0);
        assert!(matches!(
            a.is_monotonically_increasing(),
            Err(ExtremaError::EmptyArray)
        ));
        assert!(matches!(
            a.is_monotonically_decreasing(),
            Err(ExtremaError::EmptyArray)
        ));
        assert!(matches!(
            a.is_strictly_increasing(),
            Err(ExtremaError::EmptyArray)
        ));
        assert!(matches!(
            a.is_strictly_decreasing(),
            Err(ExtremaError::EmptyArray)
        ));
    }

    #[test]
    fn test_nan_propagates_as_undefined_order() {
        let a = array![1.0, f64::NAN, 2.0];

        assert!(matches!(
            a.is_monotonically_increasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
        assert!(matches!(
            a.is_monotonically_decreasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
        assert!(matches!(
            a.is_strictly_increasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
        assert!(matches!(
            a.is_strictly_decreasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
    }

    #[test]
    fn test_nan_as_first_element() {
        let a = array![f64::NAN, 1.0];

        assert!(matches!(
            a.is_monotonically_increasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
    }

    // ── minmax ──────────────────────────────────────────────────────

    #[test]
    fn test_minmax_basic() {
        let a = array![3.0, 1.0, 4.0, 1.5, 9.2, 2.6];
        assert_eq!(a.minmax(), Ok((1.0, 9.2)));
    }

    #[test]
    fn test_minmax_empty() {
        let a: Array1<f64> = Array1::from_vec(vec![]);
        assert_eq!(a.minmax(), Err(ExtremaError::EmptyArray));
    }

    #[test]
    fn test_minmax_single_element() {
        let a = array![42.0];
        assert_eq!(a.minmax(), Ok((42.0, 42.0)));
    }

    #[test]
    fn test_minmax_with_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        assert_eq!(a.minmax(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_minmax_nan_first() {
        let a = array![f64::NAN, 1.0, 2.0];
        assert_eq!(a.minmax(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_minmax_identical_elements() {
        let a = array![5.0, 5.0, 5.0, 5.0];
        assert_eq!(a.minmax(), Ok((5.0, 5.0)));
    }

    #[test]
    fn test_minmax_two_elements() {
        let a = array![10.0, 3.0];
        assert_eq!(a.minmax(), Ok((3.0, 10.0)));
    }

    #[test]
    fn test_minmax_negative_values() {
        let a = array![-5.0, -1.0, -10.0, -3.0];
        assert_eq!(a.minmax(), Ok((-10.0, -1.0)));
    }

    #[test]
    fn test_minmax_integers() {
        let a = array![3, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(a.minmax(), Ok((1, 9)));
    }

    // ── Extrema: all-identical, all-negative, large arrays ─────────

    #[test]
    fn test_maxval_minval_identical() {
        let a = array![7.0, 7.0, 7.0];
        assert_eq!(a.maxval(), Ok(7.0));
        assert_eq!(a.minval(), Ok(7.0));
    }

    #[test]
    fn test_maxval_minval_all_negative() {
        let a = array![-3.0, -1.0, -7.0, -2.0];
        assert_eq!(a.maxval(), Ok(-1.0));
        assert_eq!(a.minval(), Ok(-7.0));
    }

    #[test]
    fn test_maxval_minval_two_elements() {
        let a = array![2.0, 8.0];
        assert_eq!(a.maxval(), Ok(8.0));
        assert_eq!(a.minval(), Ok(2.0));
    }

    #[test]
    fn test_maxval_nan_at_end() {
        let a = array![1.0, 2.0, f64::NAN];
        assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_maxval_nan_as_first() {
        let a = array![f64::NAN, 1.0, 2.0];
        assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
        assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_argmax_argmin_single() {
        let a = array![99.0];
        assert_eq!(a.argmax(), Ok(0));
        assert_eq!(a.argmin(), Ok(0));
    }

    #[test]
    fn test_argmax_argmin_empty() {
        let a: Array1<f64> = Array1::from_vec(vec![]);
        assert_eq!(a.argmax(), Err(ExtremaError::EmptyArray));
        assert_eq!(a.argmin(), Err(ExtremaError::EmptyArray));
    }

    #[test]
    fn test_argmax_returns_first_occurrence() {
        let a = array![1.0, 5.0, 5.0, 3.0];
        assert_eq!(a.argmax(), Ok(1)); // first 5.0
    }

    #[test]
    fn test_argmin_returns_first_occurrence() {
        let a = array![3.0, 1.0, 1.0, 5.0];
        assert_eq!(a.argmin(), Ok(1)); // first 1.0
    }

    #[test]
    fn test_argmax_argmin_2d() {
        let a = array![[10, 20], [30, 5]];
        assert_eq!(a.argmax(), Ok((1, 0))); // 30 at (1,0)
        assert_eq!(a.argmin(), Ok((1, 1))); // 5 at (1,1)
    }

    // ── along: single-row/col, NaN in specific lanes ───────────────

    #[test]
    fn test_maxval_along_single_row() {
        let a = array![[3.0, 1.0, 4.0]];
        assert_eq!(a.maxval_along(Axis(0)).unwrap(), array![3.0, 1.0, 4.0]);
        assert_eq!(a.maxval_along(Axis(1)).unwrap(), array![4.0]);
    }

    #[test]
    fn test_minval_along_single_row() {
        let a = array![[3.0, 1.0, 4.0]];
        assert_eq!(a.minval_along(Axis(0)).unwrap(), array![3.0, 1.0, 4.0]);
        assert_eq!(a.minval_along(Axis(1)).unwrap(), array![1.0]);
    }

    #[test]
    fn test_argmax_along_single_row() {
        let a = array![[3.0, 1.0, 4.0]];
        assert_eq!(a.argmax_along(Axis(0)).unwrap(), array![0, 0, 0]);
        assert_eq!(a.argmax_along(Axis(1)).unwrap(), array![2]);
    }

    #[test]
    fn test_argmin_along_single_row() {
        let a = array![[3.0, 1.0, 4.0]];
        assert_eq!(a.argmin_along(Axis(0)).unwrap(), array![0, 0, 0]);
        assert_eq!(a.argmin_along(Axis(1)).unwrap(), array![1]);
    }

    #[test]
    fn test_argmax_along_nan_in_later_lane() {
        // NaN appears in the second row, so it's encountered during lane iteration
        let a = array![[1.0, 2.0], [f64::NAN, 4.0]];
        assert_eq!(a.argmax_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_argmin_along_nan_in_first_lane() {
        // NaN in the first row triggers error during first-lane NaN check
        let a = array![[f64::NAN, 2.0], [3.0, 4.0]];
        assert_eq!(a.argmin_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
    }

    #[test]
    fn test_along_methods_with_integers() {
        let a = array![[10, 20, 30], [40, 5, 60]];
        assert_eq!(a.maxval_along(Axis(0)).unwrap(), array![40, 20, 60]);
        assert_eq!(a.minval_along(Axis(0)).unwrap(), array![10, 5, 30]);
        assert_eq!(a.argmax_along(Axis(0)).unwrap(), array![1, 0, 1]);
        assert_eq!(a.argmin_along(Axis(0)).unwrap(), array![0, 1, 0]);
    }

    // ── trapezoid ──────────────────────────────────────────────────

    #[test]
    fn test_trapezoid_basic() {
        // Integrate y=x from 0 to 2: exact answer = 2.0
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0, 2.0];
        let result = y.trapezoid(&x);
        assert!((result - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_constant_function() {
        // Integrate y=3 from 0 to 4: exact answer = 12.0
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = y.trapezoid(&x);
        assert!((result - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_two_points() {
        // Single trapezoid: (1+3)/2 * (2-0) = 4.0
        let x = array![0.0, 2.0];
        let y = array![1.0, 3.0];
        let result = y.trapezoid(&x);
        assert!((result - 4.0).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_trapezoid_single_point() {
        // Cannot integrate with fewer than 2 points
        let x = array![1.0];
        let y = array![5.0];
        let result = y.trapezoid(&x);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_trapezoid_non_uniform_spacing() {
        // x = [0, 1, 4], y = [0, 1, 4] (y = x)
        // Panel 1: (0+1)/2 * (1-0) = 0.5
        // Panel 2: (1+4)/2 * (4-1) = 7.5
        // Total: 8.0; exact integral of x from 0..4 = 8.0
        let x = array![0.0, 1.0, 4.0];
        let y = array![0.0, 1.0, 4.0];
        let result = y.trapezoid(&x);
        assert!((result - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_negative_values() {
        // y = [-1, -2] over x = [0, 1]: (-1 + -2)/2 * 1 = -1.5
        let x = array![0.0, 1.0];
        let y = array![-1.0, -2.0];
        let result = y.trapezoid(&x);
        assert!((result - (-1.5)).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Arrays must have the same length")]
    fn test_trapezoid_mismatched_lengths() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![1.0, 2.0];
        y.trapezoid(&x);
    }

    // ── simpson ─────────────────────────────────────────────────────

    #[test]
    fn test_simpson_two_points() {
        let x = array![0.0, 1.0];
        let y = array![0.0, 1.0];
        assert_eq!(y.simpson(&x), 0.5);
    }

    #[test]
    fn test_simpson_cubic_uniform() {
        // y = x^3 with uniform spacing: composite Simpson is exact for cubics.
        let x = array![0.0, 1.0, 2.0];
        let y = array![0.0, 1.0, 8.0];
        assert_eq!(y.simpson(&x), 4.0);

        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 8.0, 27.0, 64.0];
        assert_eq!(y.simpson(&x), 64.0);
    }

    #[test]
    fn test_simpson_quadratic_even() {
        // y = x^2 with 4 points (odd intervals): Simpson's 3/8 rule is exact
        // for quadratics (and cubics).
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        assert!((y.simpson(&x) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_simpson_cubic_even() {
        // y = x^3 with 4 points: Simpson's 3/8 rule is exact for cubics.
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 8.0, 27.0];
        assert!((y.simpson(&x) - 20.25).abs() < 1e-12);
    }

    #[test]
    fn test_simpson_cubic_even_non_uniform() {
        // The 3/8 weights integrate a cubic exactly even with non-uniform
        // spacing.
        let x = array![0.0, 0.5, 1.7, 3.0];
        let y = array![0.0, 0.125, 4.913, 27.0];
        assert!((y.simpson(&x) - 20.25).abs() < 1e-12);
    }

    #[test]
    fn test_simpson_even_constant_non_uniform() {
        // All intervals must be covered exactly once by the 1/3 panels plus
        // the 3/8 tail.
        let x = array![0.0, 0.3, 1.0, 2.5, 4.0, 6.0];
        let y = array![3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        assert!((y.simpson(&x) - 18.0).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_simpson_single_point() {
        let x = array![1.0];
        let y = array![5.0];
        assert_eq!(y.simpson(&x), 0.0);
    }

    #[test]
    #[should_panic(expected = "Arrays must have the same length")]
    fn test_simpson_mismatched_lengths() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![1.0, 2.0];
        y.simpson(&x);
    }

    // ── as_real_slice ──────────────────────────────────────────────

    #[test]
    fn test_as_real_slice() {
        let a = array![1.0, 2.0, 3.0];
        let slice = a.as_real_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_as_real_slice_single() {
        let a = array![42.0];
        assert_eq!(a.as_real_slice(), &[42.0]);
    }

    // ── find_index_le / find_index_ge ──────────────────────────────

    #[test]
    fn test_find_index_le_exact_match() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(find_index_le(2.0, &arr), Some(1));
    }

    #[test]
    fn test_find_index_le_between() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(find_index_le(4.0, &arr), Some(1)); // 3.0 <= 4.0 < 5.0
    }

    #[test]
    fn test_find_index_le_empty() {
        let arr: [f64; 0] = [];
        assert_eq!(find_index_le(1.0, &arr), None);
    }

    #[test]
    fn test_find_index_le_below_min() {
        let arr = [2.0, 3.0, 4.0];
        assert_eq!(find_index_le(1.0, &arr), None);
    }

    #[test]
    fn test_find_index_le_at_last() {
        let arr = [1.0, 2.0, 3.0];
        // val >= last element returns None
        assert_eq!(find_index_le(3.0, &arr), None);
    }

    #[test]
    fn test_find_index_le_above_last() {
        let arr = [1.0, 2.0, 3.0];
        assert_eq!(find_index_le(10.0, &arr), None);
    }

    #[test]
    fn test_find_index_ge_exact_match() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(find_index_ge(2.0, &arr), Some(1));
    }

    #[test]
    fn test_find_index_ge_between() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(find_index_ge(4.0, &arr), Some(2)); // 5.0 >= 4.0
    }

    #[test]
    fn test_find_index_ge_empty() {
        let arr: [f64; 0] = [];
        assert_eq!(find_index_ge(1.0, &arr), None);
    }

    #[test]
    fn test_find_index_ge_above_max() {
        let arr = [1.0, 2.0, 3.0];
        assert_eq!(find_index_ge(10.0, &arr), None);
    }

    #[test]
    fn test_find_index_ge_at_first() {
        let arr = [1.0, 2.0, 3.0];
        // val <= first element returns None
        assert_eq!(find_index_ge(1.0, &arr), None);
    }

    #[test]
    fn test_find_index_ge_below_first() {
        let arr = [1.0, 2.0, 3.0];
        assert_eq!(find_index_ge(0.5, &arr), None);
    }

    // ── lower_bound_index / upper_bound_index ──────────────────────

    #[test]
    fn test_lower_bound_index_middle() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(lower_bound_index(4.0, &arr), 2);
    }

    #[test]
    fn test_lower_bound_index_below() {
        let arr = [1.0, 3.0, 5.0];
        assert_eq!(lower_bound_index(0.0, &arr), 0);
    }

    #[test]
    fn test_lower_bound_index_above() {
        let arr = [1.0, 3.0, 5.0];
        assert_eq!(lower_bound_index(10.0, &arr), 3);
    }

    #[test]
    fn test_lower_bound_index_empty() {
        let arr: [f64; 0] = [];
        assert_eq!(lower_bound_index(1.0, &arr), 0);
    }

    #[test]
    fn test_lower_bound_index_exact_match() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        // exact match at interior: partition_point finds first x >= 3.0
        assert_eq!(lower_bound_index(3.0, &arr), 1);
    }

    #[test]
    fn test_upper_bound_index_middle() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(upper_bound_index(4.0, &arr), 1); // 3.0 <= 4.0
    }

    #[test]
    fn test_upper_bound_index_below() {
        let arr = [1.0, 3.0, 5.0];
        assert_eq!(upper_bound_index(0.0, &arr), 0); // clamped
    }

    #[test]
    fn test_upper_bound_index_above() {
        let arr = [1.0, 3.0, 5.0];
        assert_eq!(upper_bound_index(10.0, &arr), 2); // clamped to last
    }

    #[test]
    fn test_upper_bound_index_empty() {
        let arr: [f64; 0] = [];
        assert_eq!(upper_bound_index(1.0, &arr), 0);
    }

    #[test]
    fn test_upper_bound_index_exact_match() {
        let arr = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(upper_bound_index(3.0, &arr), 1);
    }

    // ── Monotonicity: more edge cases ──────────────────────────────

    #[test]
    fn test_single_element_monotonicity() {
        let a = array![42.0];
        assert!(a.is_monotonically_increasing().unwrap());
        assert!(a.is_monotonically_decreasing().unwrap());
        assert!(a.is_strictly_increasing().unwrap());
        assert!(a.is_strictly_decreasing().unwrap());
        assert!(a.is_monotonic().unwrap());
        assert!(a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_constant_array_monotonicity() {
        let a = array![3.0, 3.0, 3.0];
        assert!(a.is_monotonically_increasing().unwrap());
        assert!(a.is_monotonically_decreasing().unwrap());
        assert!(!a.is_strictly_increasing().unwrap());
        assert!(!a.is_strictly_decreasing().unwrap());
        assert!(a.is_monotonic().unwrap());
        assert!(!a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_is_monotonic_true_increasing() {
        let a = array![1.0, 2.0, 3.0];
        assert!(a.is_monotonic().unwrap());
    }

    #[test]
    fn test_is_monotonic_true_decreasing() {
        let a = array![3.0, 2.0, 1.0];
        assert!(a.is_monotonic().unwrap());
    }

    #[test]
    fn test_is_monotonic_false() {
        let a = array![1.0, 3.0, 2.0];
        assert!(!a.is_monotonic().unwrap());
    }

    #[test]
    fn test_is_monotonic_empty() {
        let a = Array1::<f64>::zeros(0);
        assert!(matches!(a.is_monotonic(), Err(ExtremaError::EmptyArray)));
    }

    #[test]
    fn test_is_monotonic_with_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        assert!(matches!(
            a.is_monotonic(),
            Err(ExtremaError::UndefinedOrder)
        ));
    }

    #[test]
    fn test_is_strictly_monotonic_increasing() {
        let a = array![1.0, 2.0, 3.0];
        assert!(a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_is_strictly_monotonic_decreasing() {
        let a = array![3.0, 2.0, 1.0];
        assert!(a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_is_strictly_monotonic_false_due_to_equal() {
        let a = array![1.0, 2.0, 2.0, 3.0];
        assert!(!a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_is_strictly_monotonic_false_due_to_reversal() {
        let a = array![1.0, 3.0, 2.0];
        assert!(!a.is_strictly_monotonic().unwrap());
    }

    #[test]
    fn test_is_strictly_monotonic_empty() {
        let a = Array1::<f64>::zeros(0);
        assert!(matches!(
            a.is_strictly_monotonic(),
            Err(ExtremaError::EmptyArray)
        ));
    }

    #[test]
    fn test_is_strictly_monotonic_with_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        assert!(matches!(
            a.is_strictly_monotonic(),
            Err(ExtremaError::UndefinedOrder)
        ));
    }

    #[test]
    fn test_nan_at_end_monotonicity() {
        let a = array![1.0, 2.0, f64::NAN];
        assert!(matches!(
            a.is_monotonically_increasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
        // is_monotonically_decreasing early-exits at (1.0, 2.0) with Ok(false)
        // before reaching NaN — this is correct short-circuit behavior.
        assert_eq!(a.is_monotonically_decreasing(), Ok(false));
        assert!(matches!(
            a.is_strictly_increasing(),
            Err(ExtremaError::UndefinedOrder)
        ));
        // Same early-exit as is_monotonically_decreasing.
        assert_eq!(a.is_strictly_decreasing(), Ok(false));
    }

    #[test]
    fn test_two_element_monotonicity() {
        let inc = array![1.0, 2.0];
        assert!(inc.is_strictly_increasing().unwrap());
        assert!(inc.is_monotonically_increasing().unwrap());
        assert!(!inc.is_strictly_decreasing().unwrap());
        assert!(!inc.is_monotonically_decreasing().unwrap());

        let dec = array![2.0, 1.0];
        assert!(dec.is_strictly_decreasing().unwrap());
        assert!(!dec.is_strictly_increasing().unwrap());

        let eq = array![1.0, 1.0];
        assert!(!eq.is_strictly_increasing().unwrap());
        assert!(!eq.is_strictly_decreasing().unwrap());
        assert!(eq.is_monotonically_increasing().unwrap());
        assert!(eq.is_monotonically_decreasing().unwrap());
    }

    // ── Special float values ───────────────────────────────────────

    #[test]
    fn test_extrema_with_infinity() {
        let a = array![1.0, f64::INFINITY, -f64::INFINITY, 0.0];
        assert_eq!(a.maxval(), Ok(f64::INFINITY));
        assert_eq!(a.minval(), Ok(f64::NEG_INFINITY));
        assert_eq!(a.minmax(), Ok((f64::NEG_INFINITY, f64::INFINITY)));
        assert_eq!(a.argmax(), Ok(1));
        assert_eq!(a.argmin(), Ok(2));
    }

    #[test]
    fn test_monotonicity_with_infinity() {
        let a = array![-f64::INFINITY, 0.0, f64::INFINITY];
        assert!(a.is_strictly_increasing().unwrap());
        assert!(a.is_monotonically_increasing().unwrap());
    }
}
