//! FASTA support in Rust: a streaming parser, the `SequenceRecord` Python
//! class that replaces Biopython's `SeqRecord` for the reference sequence
//! and alignments, and the `FastaReader` iterator exposed as
//! `PyEvoMotionParser.read_fasta`.
//!
//! Semantics follow Biopython's FASTA reader where PyEvoMotion relied on
//! them: `id` is the first whitespace-delimited token of the header,
//! `description` is the whole header line without the leading `>`, and
//! whitespace inside sequence lines is dropped. Case is preserved.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

/// One parsed FASTA record, before it becomes a Python object.
pub struct RawRecord {
    pub id: String,
    pub description: String,
    pub seq: String,
}

/// Streaming FASTA parser over any buffered reader. Records are produced
/// one at a time, so multi-gigabyte files are never held in memory.
pub struct FastaParser<R: BufRead> {
    reader: R,
    pending_header: Option<String>,
    line: String,
}

impl<R: BufRead> FastaParser<R> {
    pub fn new(reader: R) -> Self {
        FastaParser {
            reader,
            pending_header: None,
            line: String::new(),
        }
    }

    fn read_line(&mut self) -> io::Result<Option<&str>> {
        self.line.clear();
        if self.reader.read_line(&mut self.line)? == 0 {
            return Ok(None);
        }
        Ok(Some(self.line.trim_end_matches(['\n', '\r'])))
    }

    /// Next record, or `None` at end of input. Text before the first header
    /// is ignored, like Biopython does.
    pub fn next_record(&mut self) -> io::Result<Option<RawRecord>> {
        let header = match self.pending_header.take() {
            Some(h) => h,
            None => loop {
                match self.read_line()? {
                    None => return Ok(None),
                    Some(text) => {
                        if let Some(h) = text.strip_prefix('>') {
                            break h.to_string();
                        }
                    }
                }
            },
        };

        let mut seq = String::new();
        loop {
            match self.read_line()? {
                None => break,
                Some(text) => {
                    if let Some(h) = text.strip_prefix('>') {
                        self.pending_header = Some(h.to_string());
                        break;
                    }
                    seq.extend(text.chars().filter(|c| !c.is_whitespace()));
                }
            }
        }

        let id = header.split_whitespace().next().unwrap_or("").to_string();
        Ok(Some(RawRecord {
            id,
            description: header,
            seq,
        }))
    }
}

/// Open a FASTA file for streaming.
pub fn open(path: &str) -> PyResult<FastaParser<BufReader<File>>> {
    let file = File::open(path)
        .map_err(|e| PyIOError::new_err(format!("Cannot open FASTA file {}: {}", path, e)))?;
    Ok(FastaParser::new(BufReader::new(file)))
}

fn io_err(path: &str, e: io::Error) -> PyErr {
    PyIOError::new_err(format!("Error reading FASTA file {}: {}", path, e))
}

/// Parse every record of an in-memory FASTA text.
pub fn parse_str(text: &str) -> Vec<RawRecord> {
    let mut parser = FastaParser::new(Cursor::new(text.as_bytes()));
    let mut out = Vec::new();
    while let Ok(Some(rec)) = parser.next_record() {
        out.push(rec);
    }
    out
}

/// First record of a FASTA file, or `None` if it has no records.
pub fn first_record(path: &str) -> PyResult<Option<RawRecord>> {
    open(path)?.next_record().map_err(|e| io_err(path, e))
}

/// The record whose id equals `target_id`, scanning until found.
pub fn find_by_id(path: &str, target_id: &str) -> PyResult<Option<RawRecord>> {
    let mut parser = open(path)?;
    while let Some(rec) = parser.next_record().map_err(|e| io_err(path, e))? {
        if rec.id == target_id {
            return Ok(Some(rec));
        }
    }
    Ok(None)
}

// ─────────────────────── Python classes ───────────────────────

/// A named sequence: what `PyEvoMotion.reference` holds, what
/// `PyEvoMotionParser.read_fasta` yields and what `generate_alignment`
/// returns (two aligned records). `seq` is a plain `str`, so `len()`,
/// slicing, `.count()` and `.upper()` all behave as on any string.
#[pyclass(name = "SequenceRecord", module = "PyEvoMotion")]
#[derive(Clone)]
pub struct SequenceRecord {
    /// First token of the FASTA header.
    #[pyo3(get)]
    pub id: String,
    /// Full FASTA header (without the leading ``>``).
    #[pyo3(get)]
    pub description: String,
    /// The sequence, as a string.
    #[pyo3(get)]
    pub seq: String,
}

impl From<RawRecord> for SequenceRecord {
    fn from(r: RawRecord) -> Self {
        SequenceRecord {
            id: r.id,
            description: r.description,
            seq: r.seq,
        }
    }
}

#[pymethods]
impl SequenceRecord {
    #[new]
    #[pyo3(signature = (id, seq, description=None))]
    fn new(id: String, seq: String, description: Option<String>) -> Self {
        SequenceRecord {
            description: description.unwrap_or_else(|| id.clone()),
            id,
            seq,
        }
    }

    /// Alias of ``id`` (Biopython compatibility).
    #[getter]
    fn name(&self) -> &str {
        &self.id
    }

    fn __len__(&self) -> usize {
        self.seq.len()
    }

    fn __repr__(&self) -> String {
        format!("SequenceRecord(id='{}', length={})", self.id, self.seq.len())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id && self.seq == other.seq
    }

    /// Copy with the sequence upper-cased.
    fn upper(&self) -> Self {
        SequenceRecord {
            id: self.id.clone(),
            description: self.description.clone(),
            seq: self.seq.to_ascii_uppercase(),
        }
    }

    /// Copy with the sequence lower-cased.
    fn lower(&self) -> Self {
        SequenceRecord {
            id: self.id.clone(),
            description: self.description.clone(),
            seq: self.seq.to_ascii_lowercase(),
        }
    }

    /// The record as FASTA text (``>description\\nsequence\\n``).
    fn format(&self) -> String {
        format!(">{}\n{}\n", self.description, self.seq)
    }
}

/// Lazy iterator over the records of a FASTA file
/// (``for record in PyEvoMotionParser.read_fasta(path)``).
#[pyclass(name = "FastaReader", module = "PyEvoMotion")]
pub struct FastaReader {
    path: String,
    inner: FastaParser<BufReader<File>>,
}

#[pymethods]
impl FastaReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        Ok(FastaReader {
            path: path.to_string(),
            inner: open(path)?,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<SequenceRecord>> {
        let path = slf.path.clone();
        slf.inner
            .next_record()
            .map(|r| r.map(SequenceRecord::from))
            .map_err(|e| io_err(&path, e))
    }

    fn __repr__(&self) -> String {
        format!("FastaReader({:?})", self.path)
    }
}

/// Build a `SequenceRecord` from any object exposing ``id`` and ``seq``
/// attributes (a `SequenceRecord`, or a Biopython record if a caller still
/// passes one).
pub fn record_from_any(obj: &Bound<'_, PyAny>) -> PyResult<SequenceRecord> {
    if let Ok(rec) = obj.extract::<SequenceRecord>() {
        return Ok(rec);
    }
    let id: String = obj
        .getattr("id")
        .and_then(|v| v.extract())
        .map_err(|_| PyValueError::new_err("expected a SequenceRecord (object with `id` and `seq`)"))?;
    let seq: String = obj.getattr("seq")?.call_method0("__str__")?.extract()?;
    Ok(SequenceRecord {
        description: id.clone(),
        id,
        seq,
    })
}
