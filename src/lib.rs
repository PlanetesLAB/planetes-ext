pub mod io {
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;
    use std::io::Error;
    use std::path::Path;

    /// Loads the contents of a file into a vector of strings, where each string is a line from the file.
    ///
    /// # Arguments
    /// * `path` - A path to the file to be read. Can be any type that implements `AsRef<Path>`.
    ///
    /// # Returns
    /// * `Ok(Vec<String>)` containing the lines of the file if successful.
    ///
    /// # Errors
    /// * `Err(Error)` if there was an error opening or reading the file.
    pub fn load_file_as_str<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        reader.lines().collect()
    }
}

pub mod arrays {
    use std::fmt;

    use ndarray::s;
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
        /// use prosia_extensions::arrays::ArrayExtrema;
        /// use prosia_extensions::arrays::ExtremaError;
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
        /// See [`ArrayExtrema::maxval`].
        #[inline]
        fn maxval(&self) -> Result<T, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut max_val = None;
            for &val in self {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match max_val {
                    None => max_val = Some(val),
                    Some(current_max) => {
                        match val.partial_cmp(&current_max) {
                            Some(std::cmp::Ordering::Greater) => max_val = Some(val),
                            Some(_) => {} // val <= current_max, keep current_max
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }
            Ok(max_val.unwrap()) // Safe because we checked for empty array above
        }

        /// See [`ArrayExtrema::minval`].
        #[inline]
        fn minval(&self) -> Result<T, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut min_val = None;
            for &val in self {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match min_val {
                    None => min_val = Some(val),
                    Some(current_min) => {
                        match val.partial_cmp(&current_min) {
                            Some(std::cmp::Ordering::Less) => min_val = Some(val),
                            Some(_) => {} // val >= current_min, keep current_min
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }
            Ok(min_val.unwrap()) // Safe because we checked for empty array above
        }

        /// See [`ArrayExtrema::maxval_along`].
        #[inline]
        fn maxval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<T, ExtremaError> {
                let mut max_val = None;
                for &val in subview {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match max_val {
                        None => max_val = Some(val),
                        Some(current_max) => match val.partial_cmp(&current_max) {
                            Some(std::cmp::Ordering::Greater) => max_val = Some(val),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }
                Ok(max_val.unwrap()) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in &mut result {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<T, ExtremaError> elements to T
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [`ArrayExtrema::minval_along`].
        #[inline]
        fn minval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<T, ExtremaError> {
                let mut min_val = None;
                for &val in subview {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match min_val {
                        None => min_val = Some(val),
                        Some(current_min) => match val.partial_cmp(&current_min) {
                            Some(std::cmp::Ordering::Less) => min_val = Some(val),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }
                Ok(min_val.unwrap()) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in &mut result {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<T, ExtremaError> elements to T
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [`ArrayExtrema::argmax`].
        #[inline]
        fn argmax(&self) -> Result<D::Pattern, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut best = None;

            for (idx, &val) in self.indexed_iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match best {
                    None => best = Some((idx, val)),
                    Some((_, best_val)) => {
                        match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Greater) => best = Some((idx, val)),
                            Some(_) => {} // val <= best_val, keep current best
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }

            Ok(best.unwrap().0) // Safe because we checked for empty array above
        }

        /// See [`ArrayExtrema::argmin`].
        #[inline]
        fn argmin(&self) -> Result<D::Pattern, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut best = None;

            for (idx, &val) in self.indexed_iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match best {
                    None => best = Some((idx, val)),
                    Some((_, best_val)) => {
                        match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Less) => best = Some((idx, val)),
                            Some(_) => {} // val >= best_val, keep current best
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }

            Ok(best.unwrap().0) // Safe because we checked for empty array above
        }

        /// See [`ArrayExtrema::argmax_along`].
        #[inline]
        fn argmax_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<usize, ExtremaError> {
                let mut best = None;

                for (idx, &val) in subview.indexed_iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match best {
                        None => best = Some((idx, val)),
                        Some((_, best_val)) => match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Greater) => best = Some((idx, val)),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }

                Ok(best.unwrap().0) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in &mut result {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<usize, ExtremaError> elements to usize
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [`ArrayExtrema::argmin_along`].
        #[inline]
        fn argmin_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<usize, ExtremaError> {
                let mut best = None;

                for (idx, &val) in subview.indexed_iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match best {
                        None => best = Some((idx, val)),
                        Some((_, best_val)) => match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Less) => best = Some((idx, val)),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }

                Ok(best.unwrap().0) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in &mut result {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<usize, ExtremaError> elements to usize
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
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
    }

    impl Integrable for RVector {
        #[inline]
        fn trapezoid(&self, x: &RVector) -> f64 {
            assert_eq!(self.len(), x.len(), "Arrays must have the same length");

            let y0 = self.slice(s![..-1]);
            let y1 = self.slice(s![1..]);
            let x0 = x.slice(s![..-1]);
            let x1 = x.slice(s![1..]);

            ((&y0 + &y1) / 2.0 * (&x1 - &x0)).sum()
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
    /// use prosia_extensions::arrays::find_index_le;
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
    /// use prosia_extensions::arrays::find_index_ge;
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
    /// use prosia_extensions::arrays::lower_bound_index;
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
    /// use prosia_extensions::arrays::upper_bound_index;
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
    }
}

pub mod types {
    use std::iter::Sum;
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

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
}
