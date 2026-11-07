//! Text parsing and fixed-width field helpers.

/// Parses a comma-separated list, trimming brackets, quotes, and whitespace.
///
/// Handles both plain lists such as `"[H2, CO, H2O]"` and NumPy-style entries
/// such as `"np.str_('H2')"`. Empty items are dropped.
///
/// # Examples
/// ```
/// use planetes_ext::text::parse_list;
///
/// assert_eq!(parse_list("[H2, CO, H2O]"), ["H2", "CO", "H2O"]);
/// assert_eq!(parse_list("['H2', 'CO']"), ["H2", "CO"]);
/// assert_eq!(parse_list("np.str_('H2')"), ["H2"]);
/// assert!(parse_list("[]").is_empty());
/// ```
#[must_use]
#[inline]
pub fn parse_list(s: &str) -> Vec<String> {
    let s = s.trim();
    let s = s.trim_matches(&['[', ']'][..]);

    if s.is_empty() {
        return Vec::new();
    }

    s.split(',')
        .map(|item| {
            let item = item.trim();
            if let Some(start) = item.find("np.str_(") {
                // Handle new-style single-element lists.
                let inner = &item[start + 8..];
                inner.trim_matches(&['(', ')', '\'', '"'][..]).to_string()
            } else {
                // Old-style lists.
                item.trim_matches(&['\'', '"'][..]).to_string()
            }
        })
        .filter(|item| !item.is_empty())
        .collect()
}

/// Returns the trimmed slice of `line` between `start` and `end`.
///
/// `end` is clamped to the line length. Returns `None` when the range is
/// empty, out of bounds, not aligned with UTF-8 character boundaries, or
/// contains only whitespace.
///
/// # Examples
/// ```
/// use planetes_ext::text::get_field;
///
/// let line = "H2   1.0e-4";
/// assert_eq!(get_field(line, 0, 2), Some("H2"));
/// assert_eq!(get_field(line, 5, 11), Some("1.0e-4"));
/// assert_eq!(get_field(line, 5, 100), Some("1.0e-4"));
/// assert_eq!(get_field(line, 2, 5), None);
/// ```
#[must_use]
#[inline]
pub fn get_field(line: &str, start: usize, end: usize) -> Option<&str> {
    if start >= line.len() {
        return None;
    }

    let end = end.min(line.len());
    let slice = line.get(start..end)?;
    let trimmed = slice.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::{get_field, parse_list};

    #[test]
    fn test_parse_list_plain() {
        assert_eq!(parse_list("[H2, CO, H2O]"), ["H2", "CO", "H2O"]);
    }

    #[test]
    fn test_parse_list_quoted() {
        assert_eq!(parse_list("['H2', \"CO\", 'H2O']"), ["H2", "CO", "H2O"]);
    }

    #[test]
    fn test_parse_list_numpy_style() {
        assert_eq!(parse_list("np.str_('H2')"), ["H2"]);
        assert_eq!(parse_list("[np.str_('H2'), np.str_(\"CO\")]"), ["H2", "CO"]);
    }

    #[test]
    fn test_parse_list_trims_whitespace_and_empty_items() {
        assert_eq!(parse_list("  [ H2 , , CO ]  "), ["H2", "CO"]);
    }

    #[test]
    fn test_parse_list_empty() {
        assert_eq!(parse_list(""), Vec::<String>::new());
        assert_eq!(parse_list("[]"), Vec::<String>::new());
        assert_eq!(parse_list("[ ]"), Vec::<String>::new());
    }

    #[test]
    fn test_get_field_basic() {
        let line = "H2   1.0e-4";
        assert_eq!(get_field(line, 0, 2), Some("H2"));
        assert_eq!(get_field(line, 5, 11), Some("1.0e-4"));
    }

    #[test]
    fn test_get_field_clamps_end() {
        let line = "H2   1.0e-4";
        assert_eq!(get_field(line, 5, 100), Some("1.0e-4"));
    }

    #[test]
    fn test_get_field_out_of_bounds() {
        let line = "H2   1.0e-4";
        assert_eq!(get_field(line, line.len(), line.len() + 1), None);
        assert_eq!(get_field(line, 5, 4), None);
    }

    #[test]
    fn test_get_field_whitespace_only() {
        let line = "H2   1.0e-4";
        assert_eq!(get_field(line, 2, 5), None);
    }

    #[test]
    fn test_get_field_utf8_boundaries() {
        let line = "αβγ";
        assert_eq!(get_field(line, 0, 2), Some("α"));
        assert_eq!(get_field(line, 2, 4), Some("β"));
        // Splitting a multi-byte character returns None instead of panicking.
        assert_eq!(get_field(line, 1, 3), None);
    }
}
