//! Formatting helpers for durations.

use std::time::Duration;

/// Formats a [`Duration`] as a compact human-readable string.
///
/// Durations below one second are shown in whole milliseconds, and durations
/// below one minute in seconds with one decimal place.
/// Longer durations add minutes, hours, and days as needed, always including
/// all smaller units.
///
/// # Examples
/// ```
/// use std::time::Duration;
/// use planetes_ext::time::format_duration;
///
/// assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
/// assert_eq!(format_duration(Duration::from_millis(1500)), "1.5s");
/// assert_eq!(format_duration(Duration::from_secs(61)), "1m 1s");
/// assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
/// assert_eq!(format_duration(Duration::from_secs(90061)), "1d 1h 1m 1s");
/// ```
#[must_use]
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let millis = d.subsec_millis();

    if total_secs == 0 {
        format!("{millis}ms")
    } else if total_secs < 60 {
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
        assert_eq!(format_duration(Duration::ZERO), "0ms");
    }

    #[test]
    fn test_sub_second_millis() {
        assert_eq!(format_duration(Duration::from_micros(999)), "0ms");
        assert_eq!(format_duration(Duration::from_millis(1)), "1ms");
        assert_eq!(format_duration(Duration::from_millis(99)), "99ms");
        assert_eq!(format_duration(Duration::from_millis(999)), "999ms");
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
