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
