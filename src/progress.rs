//! Plain-text progress reporting for HPC and batch jobs.
//!
//! All output consists of newline-delimited plain-text lines without ANSI
//! escape codes or carriage returns, so logs remain readable when redirected
//! to files or job-scheduler output.
//!
//! Low-level building blocks ([`Counter`], [`Snapshot`], [`Throttle`],
//! [`report_while`], [`format_count`], [`format_eta`]) allow downstream crates
//! to assemble custom reporters without duplicating atomic bookkeeping,
//! heartbeat throttling, or ETA calculation. Ready-made [`Progress`] and
//! [`Stage`] cover the standard line formats.

use std::borrow::Cow;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::{Duration, Instant};

use jiff::ZonedDateTime;

use crate::time::TIME_FORMAT;
use crate::time::format_duration;

/// Default interval between periodic progress lines.
pub const DEFAULT_HEARTBEAT: Duration = Duration::from_secs(60);

/// Formats a count compactly, e.g. `1234` as `1.2k` and `4000000` as `4M`.
///
/// Counts below `1000` are returned unchanged.
///
/// # Examples
/// ```
/// use planetes_ext::progress::format_count;
///
/// assert_eq!(format_count(42), "42");
/// assert_eq!(format_count(1_234), "1.2k");
/// assert_eq!(format_count(4_000_000), "4M");
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_count(n: usize) -> String {
    let (value, suffix) = match n {
        0..1_000 => return n.to_string(),
        1_000..999_950 => (n as f64 / 1e3, "k"),
        _ => (n as f64 / 1e6, "M"),
    };
    let s = format!("{value:.1}");
    format!("{}{suffix}", s.strip_suffix(".0").unwrap_or(&s))
}

/// Formats an estimated time remaining as whole seconds (`42s`),
/// minutes and zero-padded seconds (`3m 05s`), or hours and zero-padded
/// minutes (`2h 07m`). Returns `"unknown"` when `eta` is `None`.
///
/// # Examples
/// ```
/// use std::time::Duration;
/// use planetes_ext::progress::format_eta;
///
/// assert_eq!(format_eta(None), "unknown");
/// assert_eq!(format_eta(Some(Duration::from_secs(42))), "42s");
/// assert_eq!(format_eta(Some(Duration::from_secs(185))), "3m 05s");
/// assert_eq!(format_eta(Some(Duration::from_mins(127))), "2h 07m");
/// ```
#[must_use]
pub fn format_eta(eta: Option<Duration>) -> String {
    let Some(d) = eta else {
        return "unknown".to_owned();
    };
    let s = d.as_secs() + u64::from(d.subsec_nanos() >= 500_000_000);
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, s % 3600 / 60),
    }
}

/// A point-in-time reading of a [`Counter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// Completed items so far, including any resumed items.
    pub finished: usize,
    /// Total number of items.
    pub total: usize,
    /// Completed work units so far, including any resumed work.
    pub worked: usize,
    /// Total number of work units.
    pub work: usize,
    /// Work units already completed before this run started.
    pub initial: usize,
    /// Wall time elapsed since the counter started.
    pub elapsed: Duration,
}

impl Snapshot {
    /// Remaining work units.
    #[must_use]
    #[inline]
    pub const fn remaining_work(&self) -> usize {
        self.work.saturating_sub(self.worked)
    }

    /// Completion percentage based on work units. Returns `100.0` when `work` is zero.
    #[must_use]
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn percent(&self) -> f64 {
        if self.work == 0 {
            100.0
        } else {
            100.0 * self.worked as f64 / self.work as f64
        }
    }

    /// Estimated time remaining, computed from work completed in this run
    /// (`worked - initial`).
    ///
    /// Returns `Some(Duration::ZERO)` when no work remains, and `None` when
    /// work remains but no progress has been recorded in this run yet.
    #[must_use]
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn eta(&self) -> Option<Duration> {
        let remaining = self.remaining_work();
        if remaining == 0 {
            return Some(Duration::ZERO);
        }
        let done = self.worked.saturating_sub(self.initial);
        if done == 0 {
            return None;
        }
        let secs = remaining as f64 * self.elapsed.as_secs_f64() / done as f64;
        Duration::try_from_secs_f64(secs).ok()
    }

    /// Estimated time remaining formatted with [`format_eta`].
    #[must_use]
    #[inline]
    pub fn eta_string(&self) -> String {
        format_eta(self.eta())
    }
}

/// Lock-free atomic progress counter tracking `total` items made of `work`
/// units in all.
///
/// Updates use only `Ordering::Relaxed` atomic operations and perform no clock
/// reads, keeping hot parallel loops overhead-free.
#[derive(Debug)]
pub struct Counter {
    total: usize,
    work: usize,
    initial: usize,
    finished: AtomicUsize,
    worked: AtomicUsize,
    start: Instant,
}

impl Counter {
    /// Creates a counter over `total` items where each item is one work unit.
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self::with_work(total, total)
    }

    /// Creates a counter over `total` items composed of `work` total work units.
    #[must_use]
    pub fn with_work(total: usize, work: usize) -> Self {
        Self {
            total,
            work,
            initial: 0,
            finished: AtomicUsize::new(0),
            worked: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// Starts from `finished` items already done, e.g. on resume, for counters
    /// whose items are one work unit each. The ETA uses only this run's rate.
    #[must_use]
    pub fn resumed(mut self, finished: usize) -> Self {
        self.finished = AtomicUsize::new(finished);
        self.worked = AtomicUsize::new(finished);
        self.initial = finished;
        self
    }

    /// Records `work` more work units and `finished` more items, returning
    /// `(before_finished, after_finished)`.
    #[inline]
    pub fn add(&self, work: usize, finished: usize) -> (usize, usize) {
        if work > 0 {
            self.worked.fetch_add(work, Ordering::Relaxed);
        }
        let before = if finished > 0 {
            self.finished.fetch_add(finished, Ordering::Relaxed)
        } else {
            self.finished.load(Ordering::Relaxed)
        };
        (before, before + finished)
    }

    /// Records `n` completed items (one work unit each), returning
    /// `(before_finished, after_finished)`.
    #[inline]
    pub fn inc(&self, n: usize) -> (usize, usize) {
        self.add(n, n)
    }

    /// Wall time elapsed since this counter started.
    #[must_use]
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Takes a [`Snapshot`] of the current progress.
    #[must_use]
    #[inline]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            finished: self.finished.load(Ordering::Relaxed),
            total: self.total,
            worked: self.worked.load(Ordering::Relaxed),
            work: self.work,
            initial: self.initial,
            elapsed: self.start.elapsed(),
        }
    }

    /// Reports [`Snapshot`] via `report` at the start, every `interval` while
    /// `run` executes, and once after `run` finishes.
    pub fn report_while<T>(
        &self,
        interval: Duration,
        report: impl Fn(Snapshot) + Sync,
        run: impl FnOnce() -> T,
    ) -> T {
        report_while(interval, || report(self.snapshot()), run)
    }
}

/// Lock-free gate that triggers when finished items cross a multiple of `every`
/// or when a heartbeat interval has elapsed since the previous trigger.
#[derive(Debug)]
pub struct Throttle {
    every: Option<usize>,
    heartbeat_secs: u64,
    last: AtomicU64,
}

impl Throttle {
    /// Creates a heartbeat-only throttle firing after `interval`.
    ///
    /// A zero duration disables time-based heartbeats.
    #[must_use]
    pub const fn heartbeat(interval: Duration) -> Self {
        Self {
            every: None,
            heartbeat_secs: interval.as_secs(),
            last: AtomicU64::new(0),
        }
    }

    /// Also triggers each time finished items cross a multiple of `every`.
    #[must_use]
    pub fn every(mut self, every: usize) -> Self {
        self.every = Some(every.max(1));
        self
    }

    /// Sets the heartbeat interval, preserving any item-count interval.
    ///
    /// A zero duration disables time-based heartbeats.
    #[must_use]
    pub const fn with_heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat_secs = interval.as_secs();
        self
    }

    /// The configured heartbeat interval.
    #[must_use]
    #[inline]
    pub const fn heartbeat_interval(&self) -> Duration {
        Duration::from_secs(self.heartbeat_secs)
    }

    /// Returns `true` if a report is due given finished item counts
    /// `(before, after)` and `elapsed` time since the run began.
    #[inline]
    pub fn check(&self, before: usize, after: usize, elapsed: Duration) -> bool {
        let now = elapsed.as_secs();
        let every_due = self
            .every
            .is_some_and(|every| after / every != before / every);
        if every_due {
            self.last.store(now, Ordering::Relaxed);
            return true;
        }
        if self.heartbeat_secs == 0 {
            return false;
        }
        let last = self.last.load(Ordering::Relaxed);
        now >= last.saturating_add(self.heartbeat_secs)
            && self
                .last
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
    }
}

/// Calls `report` at the start, every `interval` while `run` executes, and
/// once after `run` finishes.
///
/// The reporting thread runs inside [`thread::scope`] and stops cleanly even
/// if `run` panics.
pub fn report_while<T>(
    interval: Duration,
    report: impl Fn() + Sync,
    run: impl FnOnce() -> T,
) -> T {
    report();
    let result = thread::scope(|scope| {
        let (stop, receiver) = mpsc::channel::<()>();
        let report_ref = &report;
        scope.spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = receiver.recv_timeout(interval) {
                report_ref();
            }
        });
        let result = run();
        drop(stop);
        result
    });
    report();
    result
}

/// Thread-safe plain-text progress reporter printing lines of the form
/// `label: finished/total unit (percent%), eta ...`.
///
/// By default, lines are printed only on a [`DEFAULT_HEARTBEAT`] interval with
/// un-abbreviated counts so quick calls stay quiet. Use [`Progress::every`] to
/// also print every `n` finished items, [`Progress::compact_counts`] for
/// `k`/`M` abbreviations, and [`Progress::resumed`] when continuing a previous
/// run.
#[derive(Debug)]
pub struct Progress {
    label: Cow<'static, str>,
    unit: Cow<'static, str>,
    counter: Counter,
    throttle: Throttle,
    compact_counts: bool,
}

impl Progress {
    /// Creates a heartbeat-only reporter over `total` items (one work unit each).
    #[must_use]
    pub fn new(
        label: impl Into<Cow<'static, str>>,
        unit: impl Into<Cow<'static, str>>,
        total: usize,
    ) -> Self {
        Self::with_work(label, unit, total, total)
    }

    /// Creates a heartbeat-only reporter over `total` items made of `work`
    /// work units in all.
    #[must_use]
    pub fn with_work(
        label: impl Into<Cow<'static, str>>,
        unit: impl Into<Cow<'static, str>>,
        total: usize,
        work: usize,
    ) -> Self {
        Self {
            label: label.into(),
            unit: unit.into(),
            counter: Counter::with_work(total, work),
            throttle: Throttle::heartbeat(DEFAULT_HEARTBEAT),
            compact_counts: false,
        }
    }

    /// Also prints a line each time finished items cross a multiple of `every`.
    #[must_use]
    pub fn every(mut self, every: usize) -> Self {
        self.throttle = self.throttle.every(every);
        self
    }

    /// Sets the heartbeat reporting interval.
    #[must_use]
    pub const fn heartbeat(mut self, interval: Duration) -> Self {
        self.throttle = self.throttle.with_heartbeat(interval);
        self
    }

    /// Abbreviates finished and total counts with [`format_count`].
    #[must_use]
    pub const fn compact_counts(mut self) -> Self {
        self.compact_counts = true;
        self
    }

    /// Starts from `finished` items already done, e.g. on resume, for counters
    /// whose items are one work unit each. The ETA uses only this run's rate.
    #[must_use]
    pub fn resumed(mut self, finished: usize) -> Self {
        self.counter = self.counter.resumed(finished);
        self
    }

    /// Records `work` more units and `finished` more items, printing a line if
    /// the throttle fires.
    #[inline]
    pub fn add(&self, work: usize, finished: usize) {
        let worked = self.counter.worked.fetch_add(work, Ordering::Relaxed) + work;
        let before = self.counter.finished.fetch_add(finished, Ordering::Relaxed);
        let after = before + finished;
        let elapsed = self.counter.start.elapsed();
        if !self.throttle.check(before, after, elapsed) {
            return;
        }
        let snap = Snapshot {
            finished: after,
            total: self.counter.total,
            worked,
            work: self.counter.work,
            initial: self.counter.initial,
            elapsed,
        };
        println!("{}", self.format_line(snap));
    }

    /// Records `n` completed items (one work unit each), printing a line if
    /// the throttle fires.
    #[inline]
    pub fn inc(&self, n: usize) {
        self.add(n, n);
    }

    /// Formats a progress line for `snap` without printing it.
    #[must_use]
    pub fn format_line(&self, snap: Snapshot) -> String {
        let eta = snap.eta_string();
        if self.compact_counts {
            format!(
                "{}: {}/{} {} ({:.0}%), eta {eta}",
                self.label,
                format_count(snap.finished),
                format_count(snap.total),
                self.unit,
                snap.percent(),
            )
        } else {
            format!(
                "{}: {}/{} {} ({:.0}%), eta {eta}",
                self.label,
                snap.finished,
                snap.total,
                self.unit,
                snap.percent(),
            )
        }
    }

    /// Returns a reference to the underlying [`Counter`].
    #[must_use]
    #[inline]
    pub const fn counter(&self) -> &Counter {
        &self.counter
    }

    /// Takes a [`Snapshot`] of the current progress.
    #[must_use]
    #[inline]
    pub fn snapshot(&self) -> Snapshot {
        self.counter.snapshot()
    }

    /// Prints this reporter's line at the start, on its heartbeat interval
    /// while `run` executes, and once at the end.
    pub fn report_while<T>(&self, run: impl FnOnce() -> T) -> T {
        self.counter.report_while(
            self.throttle.heartbeat_interval(),
            |snap| println!("{}", self.format_line(snap)),
            run,
        )
    }
}

/// A timed stage logged as aligned `# stage` start and end lines with the local
/// time:
///
/// ```text
/// # stage  iter 1/6    mcrt                      start  2000-01-01 12:00:00 UTC +00:00
/// # stage  iter 1/6    mcrt                      end    2000-01-01 12:04:38 UTC +00:00  4m 38s
/// ```
#[derive(Debug)]
pub struct Stage {
    context: String,
    name: String,
    start: Instant,
}

impl Stage {
    /// Prints the start line of stage `name` with `"-"` in the context column.
    #[must_use]
    pub fn start(name: impl Into<String>) -> Self {
        Self::with_context("-", name)
    }

    /// Prints the start line of stage `name` inside iteration `iter` of `niter`
    /// (`"iter {iter}/{niter}"`).
    #[must_use]
    pub fn iteration(iter: usize, niter: usize, name: impl Into<String>) -> Self {
        Self::with_context(format!("iter {iter}/{niter}"), name)
    }

    /// Prints the start line of stage `name` with an arbitrary `context` column.
    #[must_use]
    pub fn with_context(context: impl Into<String>, name: impl Into<String>) -> Self {
        let context = context.into();
        let name = name.into();
        println!(
            "# stage  {context:<11} {name:<25} start  {}",
            ZonedDateTime::now().strftime(TIME_FORMAT)
        );
        Self {
            context,
            name,
            start: Instant::now(),
        }
    }

    /// Wall time elapsed since this stage started.
    #[must_use]
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Prints the end line with the elapsed duration of the stage.
    pub fn end(self) {
        println!(
            "# stage  {:<11} {:<25} end    {}  {}",
            self.context,
            self.name,
            ZonedDateTime::now().strftime(TIME_FORMAT),
            format_duration(self.start.elapsed())
        );
    }
}

#[cfg(test)]
mod tests {
    use std::panic;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

    use super::{Counter, Progress, Snapshot, Throttle};
    use super::{format_count, format_eta, report_while};

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1k");
        assert_eq!(format_count(1_234), "1.2k");
        assert_eq!(format_count(999_949), "999.9k");
        assert_eq!(format_count(999_950), "1M");
        assert_eq!(format_count(1_500_000), "1.5M");
        assert_eq!(format_count(4_000_000), "4M");
    }

    #[test]
    fn test_format_eta() {
        assert_eq!(format_eta(None), "unknown");
        assert_eq!(format_eta(Some(Duration::ZERO)), "0s");
        assert_eq!(format_eta(Some(Duration::from_millis(499))), "0s");
        assert_eq!(format_eta(Some(Duration::from_millis(500))), "1s");
        assert_eq!(format_eta(Some(Duration::from_secs(42))), "42s");
        assert_eq!(format_eta(Some(Duration::from_millis(59_600))), "1m 00s");
        assert_eq!(format_eta(Some(Duration::from_secs(185))), "3m 05s");
        assert_eq!(format_eta(Some(Duration::from_secs(3_599))), "59m 59s");
        assert_eq!(format_eta(Some(Duration::from_secs(3_600))), "1h 00m");
        assert_eq!(format_eta(Some(Duration::from_mins(127))), "2h 07m");
    }

    #[test]
    fn test_snapshot_percent_and_eta() {
        let zero_work = Snapshot {
            finished: 0,
            total: 0,
            worked: 0,
            work: 0,
            initial: 0,
            elapsed: Duration::from_secs(1),
        };
        assert!((zero_work.percent() - 100.0).abs() < 1e-12);
        assert_eq!(zero_work.eta(), Some(Duration::ZERO));
        assert_eq!(zero_work.eta_string(), "0s");

        let no_progress = Snapshot {
            finished: 0,
            total: 100,
            worked: 0,
            work: 100,
            initial: 0,
            elapsed: Duration::from_secs(10),
        };
        assert_eq!(no_progress.eta(), None);
        assert_eq!(no_progress.eta_string(), "unknown");

        let resumed = Snapshot {
            finished: 60,
            total: 100,
            worked: 60,
            work: 100,
            initial: 50,
            elapsed: Duration::from_secs(10),
        };
        assert!((resumed.percent() - 60.0).abs() < 1e-12);
        // 10 units done in 10s -> 40 remaining take 40s.
        assert_eq!(resumed.eta(), Some(Duration::from_secs(40)));
        assert_eq!(resumed.eta_string(), "40s");
    }

    #[test]
    fn test_counter_and_resumed() {
        let counter = Counter::new(100).resumed(40);
        assert_eq!(counter.inc(5), (40, 45));
        let snap = counter.snapshot();
        assert_eq!(snap.finished, 45);
        assert_eq!(snap.worked, 45);
        assert_eq!(snap.initial, 40);
        assert_eq!(snap.remaining_work(), 55);
    }

    #[test]
    fn test_counter_dual_work() {
        let counter = Counter::with_work(10, 50);
        assert_eq!(counter.add(1, 0), (0, 0));
        assert_eq!(counter.add(1, 1), (0, 1));
        let snap = counter.snapshot();
        assert_eq!(snap.finished, 1);
        assert_eq!(snap.worked, 2);
        assert!((snap.percent() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_throttle_heartbeat_only() {
        let throttle = Throttle::heartbeat(Duration::from_secs(60));
        assert!(!throttle.check(0, 1, Duration::from_secs(59)));
        assert!(throttle.check(1, 2, Duration::from_secs(60)));
        assert!(!throttle.check(2, 3, Duration::from_secs(119)));
        assert!(throttle.check(3, 4, Duration::from_secs(120)));
    }

    #[test]
    fn test_throttle_every_resets_heartbeat() {
        let throttle = Throttle::heartbeat(Duration::from_secs(60)).every(10);
        assert!(!throttle.check(0, 9, Duration::from_secs(50)));
        // Crossing 10 fires and resets last to 50s.
        assert!(throttle.check(9, 10, Duration::from_secs(50)));
        // 60s total is only 10s since last report, so heartbeat does not fire yet.
        assert!(!throttle.check(10, 11, Duration::from_secs(60)));
        // 110s is 60s after 50s -> heartbeat fires.
        assert!(throttle.check(11, 12, Duration::from_secs(110)));
    }

    #[test]
    fn test_progress_format_line_raw_and_compact() {
        let snap = Snapshot {
            finished: 1_234,
            total: 4_000_000,
            worked: 2_000_000,
            work: 4_000_000,
            initial: 0,
            elapsed: Duration::from_secs(185),
        };
        let raw = Progress::new("kappa", "wavelengths", 4_000_000);
        assert_eq!(
            raw.format_line(snap),
            "kappa: 1234/4000000 wavelengths (50%), eta 3m 05s"
        );
        let compact = Progress::new("mcrt", "photons", 4_000_000).compact_counts();
        assert_eq!(
            compact.format_line(snap),
            "mcrt: 1.2k/4M photons (50%), eta 3m 05s"
        );
    }

    #[test]
    fn test_report_while_start_tick_end() {
        let ticks = AtomicUsize::new(0);
        let value = report_while(
            Duration::from_millis(20),
            || {
                ticks.fetch_add(1, Ordering::Relaxed);
            },
            || {
                thread::sleep(Duration::from_millis(55));
                7
            },
        );
        assert_eq!(value, 7);
        // At least start + 1 periodic tick + end.
        assert!(ticks.load(Ordering::Relaxed) >= 3);
    }

    #[test]
    fn test_report_while_stops_on_panic() {
        let ticks = AtomicUsize::new(0);
        let result = panic::catch_unwind(|| {
            report_while(
                Duration::from_secs(60),
                || {
                    ticks.fetch_add(1, Ordering::Relaxed);
                },
                || -> () {
                    panic!("boom");
                },
            );
        });
        assert!(result.is_err());
        // Start report ran, background thread exited without waiting 60s, end report skipped.
        assert_eq!(ticks.load(Ordering::Relaxed), 1);
    }
}
