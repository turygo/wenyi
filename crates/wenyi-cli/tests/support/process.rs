use std::ffi::OsString;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const CAPTURE_LIMIT: usize = 1024 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(20);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputStream {
    Stdout,
    Stderr,
}

#[derive(Debug)]
pub(crate) struct ProcessResult {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

#[derive(Debug)]
pub(crate) enum ProcessError {
    Spawn(io::Error),
    MissingPipe {
        stream: &'static str,
        child_pid: u32,
        reaped: bool,
        cleanup_error: Option<String>,
    },
    Timeout {
        child_pid: u32,
        reaped: bool,
        cleanup_error: Option<String>,
    },
    OutputLimit {
        stream: OutputStream,
        child_pid: u32,
        reaped: bool,
        cleanup_error: Option<String>,
    },
    Capture {
        stream: OutputStream,
        child_pid: u32,
        reaped: bool,
        message: String,
        cleanup_error: Option<String>,
    },
    Wait {
        child_pid: u32,
        message: String,
        reaped: bool,
        cleanup_error: Option<String>,
    },
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(error) => write!(formatter, "failed to spawn wenyi: {error}"),
            Self::MissingPipe {
                stream,
                child_pid,
                reaped,
                cleanup_error,
            } => write!(
                formatter,
                "missing {stream} pipe for child {child_pid}; reaped={reaped}; cleanup={cleanup_error:?}"
            ),
            Self::Timeout {
                child_pid,
                reaped,
                cleanup_error,
            } => write!(
                formatter,
                "child {child_pid} timed out; reaped={reaped}; cleanup={cleanup_error:?}"
            ),
            Self::OutputLimit {
                stream,
                child_pid,
                reaped,
                cleanup_error,
            } => write!(
                formatter,
                "child {child_pid} exceeded {stream:?} capture limit; reaped={reaped}; cleanup={cleanup_error:?}"
            ),
            Self::Capture {
                stream,
                child_pid,
                reaped,
                message,
                cleanup_error,
            } => write!(
                formatter,
                "child {child_pid} {stream:?} capture failed: {message}; reaped={reaped}; cleanup={cleanup_error:?}"
            ),
            Self::Wait {
                child_pid,
                message,
                reaped,
                cleanup_error,
            } => write!(
                formatter,
                "child {child_pid} wait failed: {message}; reaped={reaped}; cleanup={cleanup_error:?}"
            ),
        }
    }
}

impl std::error::Error for ProcessError {}

#[derive(Debug)]
enum CaptureFailure {
    Limit,
    Io(String),
}

#[derive(Debug)]
enum CaptureEvent {
    Limit(OutputStream),
    Io(OutputStream, String),
}

pub(crate) fn run_wenyi(
    args: &[OsString],
    cwd: &Path,
    deadline: Duration,
) -> Result<ProcessResult, ProcessError> {
    let executable = env!("CARGO_BIN_EXE_wenyi");
    let mut child = Command::new(executable)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(ProcessError::Spawn)?;
    let child_pid = child.id();
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => return Err(missing_pipe(&mut child, child_pid, "stdout")),
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => return Err(missing_pipe(&mut child, child_pid, "stderr")),
    };
    let (event_sender, events) = mpsc::channel();
    let stdout_reader = spawn_reader(stdout, OutputStream::Stdout, event_sender.clone());
    let stderr_reader = spawn_reader(stderr, OutputStream::Stderr, event_sender.clone());
    let _sender_guard = event_sender;
    let started = Instant::now();
    let deadline_at = started.checked_add(deadline);

    if deadline.is_zero() || deadline_at.is_none() {
        return Err(timeout_child(
            &mut child,
            child_pid,
            stdout_reader,
            stderr_reader,
        ));
    }
    let deadline_at = deadline_at.expect("checked above");

    loop {
        let now = Instant::now();
        if now >= deadline_at {
            return Err(timeout_child(
                &mut child,
                child_pid,
                stdout_reader,
                stderr_reader,
            ));
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                return finish_capture(status, child_pid, stdout_reader, stderr_reader);
            }
            Ok(None) => {}
            Err(error) => {
                let cleanup_error = kill_and_reap(&mut child).err();
                let _ = join_reader(stdout_reader, OutputStream::Stdout);
                let _ = join_reader(stderr_reader, OutputStream::Stderr);
                return Err(ProcessError::Wait {
                    child_pid,
                    message: error.to_string(),
                    reaped: cleanup_error.is_none(),
                    cleanup_error,
                });
            }
        }

        let remaining = deadline_at.saturating_duration_since(Instant::now());
        match events.recv_timeout(remaining.min(POLL_INTERVAL)) {
            Ok(CaptureEvent::Limit(stream)) => {
                let cleanup_error = kill_and_reap(&mut child).err();
                let _ = join_reader(stdout_reader, OutputStream::Stdout);
                let _ = join_reader(stderr_reader, OutputStream::Stderr);
                return Err(ProcessError::OutputLimit {
                    stream,
                    child_pid,
                    reaped: cleanup_error.is_none(),
                    cleanup_error,
                });
            }
            Ok(CaptureEvent::Io(stream, message)) => {
                let cleanup_error = kill_and_reap(&mut child).err();
                let _ = join_reader(stdout_reader, OutputStream::Stdout);
                let _ = join_reader(stderr_reader, OutputStream::Stderr);
                return Err(ProcessError::Capture {
                    stream,
                    child_pid,
                    reaped: cleanup_error.is_none(),
                    message,
                    cleanup_error,
                });
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => unreachable!("sender guard is retained"),
        }
    }
}

fn missing_pipe(child: &mut Child, child_pid: u32, stream: &'static str) -> ProcessError {
    let cleanup_error = kill_and_reap(child).err();
    ProcessError::MissingPipe {
        stream,
        child_pid,
        reaped: cleanup_error.is_none(),
        cleanup_error,
    }
}

fn spawn_reader<R: Read + Send + 'static>(
    reader: R,
    stream: OutputStream,
    events: Sender<CaptureEvent>,
) -> JoinHandle<Result<Vec<u8>, CaptureFailure>> {
    thread::spawn(move || {
        let result = read_bounded(reader);
        match &result {
            Err(CaptureFailure::Limit) => {
                let _ = events.send(CaptureEvent::Limit(stream));
            }
            Err(CaptureFailure::Io(message)) => {
                let _ = events.send(CaptureEvent::Io(stream, message.clone()));
            }
            Ok(_) => {}
        }
        result
    })
}

fn read_bounded<R: Read>(mut reader: R) -> Result<Vec<u8>, CaptureFailure> {
    let mut captured = Vec::with_capacity(16 * 1024);
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| CaptureFailure::Io(error.to_string()))?;
        if count == 0 {
            return Ok(captured);
        }
        if count > CAPTURE_LIMIT - captured.len() {
            return Err(CaptureFailure::Limit);
        }
        captured.extend_from_slice(&buffer[..count]);
    }
}

fn finish_capture(
    status: ExitStatus,
    child_pid: u32,
    stdout_reader: JoinHandle<Result<Vec<u8>, CaptureFailure>>,
    stderr_reader: JoinHandle<Result<Vec<u8>, CaptureFailure>>,
) -> Result<ProcessResult, ProcessError> {
    let stdout_result = join_reader(stdout_reader, OutputStream::Stdout);
    let stderr_result = join_reader(stderr_reader, OutputStream::Stderr);
    let stdout = stdout_result
        .map_err(|failure| capture_error(failure, OutputStream::Stdout, child_pid, true, None))?;
    let stderr = stderr_result
        .map_err(|failure| capture_error(failure, OutputStream::Stderr, child_pid, true, None))?;
    Ok(ProcessResult {
        status,
        stdout,
        stderr,
    })
}

fn join_reader(
    reader: JoinHandle<Result<Vec<u8>, CaptureFailure>>,
    stream: OutputStream,
) -> Result<Vec<u8>, CaptureFailure> {
    reader
        .join()
        .map_err(|_| CaptureFailure::Io(format!("{stream:?} reader panicked")))?
}

fn capture_error(
    failure: CaptureFailure,
    stream: OutputStream,
    child_pid: u32,
    reaped: bool,
    cleanup_error: Option<String>,
) -> ProcessError {
    match failure {
        CaptureFailure::Limit => ProcessError::OutputLimit {
            stream,
            child_pid,
            reaped,
            cleanup_error,
        },
        CaptureFailure::Io(message) => ProcessError::Capture {
            stream,
            child_pid,
            reaped,
            message,
            cleanup_error,
        },
    }
}

fn timeout_child(
    child: &mut Child,
    child_pid: u32,
    stdout_reader: JoinHandle<Result<Vec<u8>, CaptureFailure>>,
    stderr_reader: JoinHandle<Result<Vec<u8>, CaptureFailure>>,
) -> ProcessError {
    let cleanup_error = kill_and_reap(child).err();
    let _ = join_reader(stdout_reader, OutputStream::Stdout);
    let _ = join_reader(stderr_reader, OutputStream::Stderr);
    ProcessError::Timeout {
        child_pid,
        reaped: cleanup_error.is_none(),
        cleanup_error,
    }
}

fn kill_and_reap(child: &mut Child) -> Result<(), String> {
    let kill_error = child.kill().err();
    child
        .wait()
        .map(|_| ())
        .map_err(|wait_error| match kill_error {
            Some(kill_error) => format!("kill failed ({kill_error}); wait failed ({wait_error})"),
            None => format!("wait failed ({wait_error})"),
        })
}

#[cfg(test)]
mod tests {
    use super::{CAPTURE_LIMIT, CaptureFailure, read_bounded};
    use std::io::Cursor;

    #[test]
    fn capture_reader_rejects_output_over_the_limit() {
        let oversized = vec![0_u8; CAPTURE_LIMIT + 1];
        assert!(matches!(
            read_bounded(Cursor::new(oversized)),
            Err(CaptureFailure::Limit)
        ));
    }
}
