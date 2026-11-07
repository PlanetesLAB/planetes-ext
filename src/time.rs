//! Formatting helpers for durations.

use std::time::Duration;

/// Formats a [`Duration`] as a compact human-readable string.
///
/// Durations below one minute are shown in seconds with one decimal place.
/// Longer durations add minutes, hours, and days as needed, always including
/// all smaller units.
///
/// # Examples
/// ```
/// use std::time::Duration;
/// use planetes_ext::time::format_duration;
///
/// assert_eq!(format_duration(Duration::from_millis(500)), "0.5s");
/// assert_eq!(format_duration(Duration::from_secs(61)), "1m 1s");
/// assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
/// assert_eq!(format_duration(Duration::from_secs(90061)), "1d 1h 1m 1s");
/// ```
#[must_use]
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let millis = d.subsec_millis();

    if total_secs < 60 {
        format!("{total_secs}.{}s", millis / 100)
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{mins}m {secs}s")
    } else if total_secs < 86400 {
        let hrs = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{hrs}h {mins}m {secs}s")
    } else {
        let days = total_secs / 86400;
        let hrs = (total_secs % 86400) / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{days}d {hrs}h {mins}m {secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::format_duration;
    use std::time::Duration;

    #[test]
    fn test_zero() {
        assert_eq!(format_duration(Duration::ZERO), "0.0s");
    }

    #[test]
    fn test_sub_second_truncates_to_tenths() {
        assert_eq!(format_duration(Duration::from_millis(99)), "0.0s");
        assert_eq!(format_duration(Duration::from_millis(100)), "0.1s");
        assert_eq!(format_duration(Duration::from_millis(599)), "0.5s");
        assert_eq!(format_duration(Duration::from_millis(999)), "0.9s");
    }

    #[test]
    fn test_seconds() {
        assert_eq!(format_duration(Duration::from_secs(1)), "1.0s");
        assert_eq!(format_duration(Duration::from_secs(59)), "59.0s");
        assert_eq!(format_duration(Duration::from_millis(59_999)), "59.9s");
    }

    #[test]
    fn test_minutes() {
        assert_eq!(format_duration(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_duration(Duration::from_secs(61)), "1m 1s");
        assert_eq!(format_duration(Duration::from_secs(3599)), "59m 59s");
    }

    #[test]
    fn test_hours() {
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h 0m 0s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
        assert_eq!(format_duration(Duration::from_secs(86_399)), "23h 59m 59s");
    }

    #[test]
    fn test_days() {
        assert_eq!(format_duration(Duration::from_hours(24)), "1d 0h 0m 0s");
        assert_eq!(format_duration(Duration::from_secs(90_061)), "1d 1h 1m 1s");
        assert_eq!(
            format_duration(Duration::from_secs(2 * 86_400 + 3 * 3600 + 4 * 60 + 5)),
            "2d 3h 4m 5s"
        );
    }
}
