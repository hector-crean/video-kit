use crate::driver::{Driver, DriverError, StreamOperation};
use crate::{Sequentise, sources::{Source, Sink}};
use std::ops::Range;
use std::path::Path;
use std::process::Command;

/// FFmpeg-specific file source
#[derive(Debug, Clone)]
pub struct FFmpegFileSource {
    pub path: String,
}

impl FFmpegFileSource {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Source for FFmpegFileSource {
    fn validate(&self) -> Result<(), DriverError> {
        if Path::new(&self.path).exists() {
            Ok(())
        } else {
            Err(DriverError::Execution(format!("Input file not found: {}", self.path)))
        }
    }
    
    fn description(&self) -> String {
        format!("FFmpeg File Source: {}", self.path)
    }
}

/// FFmpeg-specific file sink
#[derive(Debug, Clone)]
pub struct FFmpegFileSink {
    pub path: String,
}

impl FFmpegFileSink {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl Sink for FFmpegFileSink {
    fn validate(&self) -> Result<(), DriverError> {
        if let Some(parent) = Path::new(&self.path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        }
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("FFmpeg File Sink: {}", self.path)
    }
}

/// FFmpeg-specific directory sink for frame extraction
#[derive(Debug, Clone)]
pub struct FFmpegDirectorySink {
    pub path: String,
    pub pattern: String,
}

impl FFmpegDirectorySink {
    pub fn new(path: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pattern: pattern.into(),
        }
    }
}

impl Sink for FFmpegDirectorySink {
    fn validate(&self) -> Result<(), DriverError> {
        std::fs::create_dir_all(&self.path)
            .map_err(|e| DriverError::Execution(format!("Cannot create output directory: {}", e)))?;
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("FFmpeg Directory Sink: {} (pattern: {})", self.path, self.pattern)
    }
}

/// Internal representation of an FFmpeg command/stream
/// This represents the operations applied but not yet executed
#[derive(Debug, Clone)]
pub struct FFmpegStream {
    pub operations: Vec<StreamOperation>,
    pub input_file: Option<String>,
}

impl FFmpegStream {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
            input_file: None,
        }
    }
    
    fn with_input(mut self, input: String) -> Self {
        self.input_file = Some(input);
        self
    }
    
    fn with_operation(mut self, op: StreamOperation) -> Self {
        self.operations.push(op);
        self
    }
}

/// FFmpeg driver implementation
pub struct FFmpegDriver {
    ffmpeg_path: String,
}

impl FFmpegDriver {
    pub fn new() -> Result<Self, DriverError> {
        // Check if ffmpeg is available in PATH
        let ffmpeg_path = Self::find_ffmpeg_executable()?;
        
        Ok(Self {
            ffmpeg_path,
        })
    }
    
    pub fn with_path(ffmpeg_path: impl Into<String>) -> Result<Self, DriverError> {
        let path = ffmpeg_path.into();
        
        // Verify the provided path is valid
        if !Path::new(&path).exists() {
            return Err(DriverError::Initialization(format!("FFmpeg executable not found at: {}", path)));
        }
        
        Ok(Self {
            ffmpeg_path: path,
        })
    }
    
    fn find_ffmpeg_executable() -> Result<String, DriverError> {
        // Try to find ffmpeg in PATH
        if let Ok(output) = Command::new("which").arg("ffmpeg").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(path);
                }
            }
        }
        
        // Try common locations
        let common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "ffmpeg", // Fallback to PATH
        ];
        
        for path in &common_paths {
            if Path::new(path).exists() || *path == "ffmpeg" {
                return Ok(path.to_string());
            }
        }
        
        Err(DriverError::Initialization(
            "FFmpeg executable not found. Please install FFmpeg or specify the path.".to_string()
        ))
    }
    
    fn build_ffmpeg_command(&self, stream: &FFmpegStream, output_path: &str) -> Result<Command, DriverError> {
        let input_file = stream.input_file.as_ref()
            .ok_or_else(|| DriverError::Execution("No input file specified".to_string()))?;

        // Determine the primary operation type
        let operation = stream.operations.iter()
            .find(|op| !matches!(op, StreamOperation::Load { .. }))
            .cloned();

        match operation {
            Some(StreamOperation::Splice { segments }) => {
                self.build_splice_command(input_file, output_path, &segments)
            }
            Some(StreamOperation::ExtractFrames { start_time, end_time, config }) => {
                self.build_extract_frames_command(input_file, output_path, start_time, end_time, config.as_ref())
            }
            Some(StreamOperation::Reverse) => {
                self.build_reverse_command(input_file, output_path)
            }
            Some(StreamOperation::CreateLoop) => {
                self.build_loop_command(input_file, output_path)
            }
            _ => {
                self.build_simple_copy_command(input_file, output_path)
            }
        }
    }

    fn build_simple_copy_command(&self, input: &str, output: &str) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args(["-i", input, "-c", "copy", "-y", output]);
        Ok(cmd)
    }

    fn build_splice_command(&self, input: &str, output: &str, segments: &[Range<f64>]) -> Result<Command, DriverError> {
        if segments.is_empty() {
            return self.build_simple_copy_command(input, output);
        }
        
        if segments.len() == 1 {
            // Single segment - simple extraction
            let segment = &segments[0];
            let mut cmd = Command::new(&self.ffmpeg_path);
            cmd.args([
                "-i", input,
                "-ss", &segment.start.to_string(),
                "-t", &(segment.end - segment.start).to_string(),
                "-c", "copy",
                "-y", output
            ]);
            Ok(cmd)
        } else {
            // Multiple segments - concatenate them
            self.build_concat_command(input, output, segments)
        }
    }
    
    fn build_poster_command(&self, input: &str, output: &str, timestamp: f64) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", input,
            "-ss", &timestamp.to_string(),
            "-vframes", "1",
            "-q:v", "2",
            "-y", output
        ]);
        Ok(cmd)
    }

    fn build_concat_command(&self, input: &str, output: &str, segments: &[Range<f64>]) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        
        // Add input file
        cmd.args(["-i", input]);
        
        // Build filter_complex for concatenation
        let mut filter_parts = Vec::new();
        
        for (i, segment) in segments.iter().enumerate() {
            let duration = segment.end - segment.start;
            filter_parts.push(format!(
                "[0:v]trim=start={}:duration={},setpts=PTS-STARTPTS[v{}];[0:a]atrim=start={}:duration={},asetpts=PTS-STARTPTS[a{}]",
                segment.start, duration, i, segment.start, duration, i
            ));
        }
        
        // Add concatenation part
        let video_inputs: String = (0..segments.len()).map(|i| format!("[v{}]", i)).collect::<Vec<_>>().join("");
        let audio_inputs: String = (0..segments.len()).map(|i| format!("[a{}]", i)).collect::<Vec<_>>().join("");
        
        filter_parts.push(format!(
            "{}concat=n={}:v=1:a=1[outv][outa]",
            video_inputs, segments.len()
        ));
        
        let filter_complex = filter_parts.join(";");
        
        cmd.args([
            "-filter_complex", &filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y", output
        ]);
        
        Ok(cmd)
    }

    fn build_extract_frames_command(&self, input: &str, output: &str, start: f64, end: f64, config: Option<&Sequentise>) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        let fps = config.and_then(|c| c.fps).unwrap_or(25.0);
        let quality = config.and_then(|c| c.quality).unwrap_or(2);

        // Extract frames to directory with pattern
        let output_pattern = format!("{}_%05d.png", output.trim_end_matches(".mp4"));
        
        cmd.args([
            "-i", input,
            "-ss", &start.to_string(),
            "-t", &(end - start).to_string(),
            "-vf", &format!("fps={}", fps),
            "-q:v", &quality.to_string(),
            "-y", &output_pattern
        ]);
        Ok(cmd)
    }

    fn build_reverse_command(&self, input: &str, output: &str) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", input,
            "-vf", "reverse",
            "-af", "areverse",
            "-y", output
        ]);
        Ok(cmd)
    }

    fn build_loop_command(&self, input: &str, output: &str) -> Result<Command, DriverError> {
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.args([
            "-i", input,
            "-filter_complex", "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[out]",
            "-map", "[out]",
            "-y", output
        ]);
        Ok(cmd)
    }
    
    fn execute_command(&self, mut cmd: Command) -> Result<(), DriverError> {
        println!("Executing FFmpeg command: {:?}", cmd);
        
        // Clear DYLD_LIBRARY_PATH to avoid conflicts with GStreamer framework libraries
        cmd.env_remove("DYLD_LIBRARY_PATH");
        
        let output = cmd.output()
            .map_err(|e| DriverError::Execution(format!("Failed to execute FFmpeg: {}", e)))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DriverError::Execution(format!("FFmpeg execution failed: {}", stderr)));
        }
        
        println!("FFmpeg command executed successfully");
        Ok(())
    }
}

impl Driver for FFmpegDriver {
    type Source = FFmpegFileSource;
    type Sink = FFmpegFileSink;
    type Stream = FFmpegStream;

    fn load(&self, source: &Self::Source) -> Result<Self::Stream, DriverError> {
        source.validate()?;
        Ok(FFmpegStream::new()
            .with_input(source.path.clone())
            .with_operation(StreamOperation::Load { source_path: source.path.clone() }))
    }

    fn splice(&self, stream: Self::Stream, segments: &[Range<f64>]) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::Splice { segments: segments.to_vec() }))
    }

    fn extract_frames(&self, stream: Self::Stream, sequentise: &Sequentise) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::ExtractFrames { start_time: sequentise.period.start, end_time: sequentise.period.end, config: Some(sequentise.clone()) }))
    }

    fn reverse(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::Reverse))
    }

    fn create_loop(&self, stream: Self::Stream) -> Result<Self::Stream, DriverError> {
        Ok(stream.with_operation(StreamOperation::CreateLoop))
    }

    fn save(&self, stream: Self::Stream, sink: &Self::Sink) -> Result<(), DriverError> {
        sink.validate()?;
        
        // Check if this is a splice operation to generate poster
        let is_splice_operation = stream.operations.iter()
            .any(|op| matches!(op, StreamOperation::Splice { .. }));
        
        if is_splice_operation {
            // Generate both video and poster for splice operations
            let cmd = self.build_ffmpeg_command(&stream, &sink.path)?;
            self.execute_command(cmd)?;
            
            // Generate poster path by replacing video extension with _poster.png
            let poster_path = if sink.path.ends_with(".mp4") {
                sink.path.replace(".mp4", "_poster.png")
            } else if sink.path.ends_with(".mov") {
                sink.path.replace(".mov", "_poster.png")
            } else if sink.path.ends_with(".avi") {
                sink.path.replace(".avi", "_poster.png")
            } else {
                format!("{}_poster.png", sink.path)
            };
            
            // Get the first segment timestamp for poster generation
            if let Some(StreamOperation::Splice { segments }) = stream.operations.iter()
                .find(|op| matches!(op, StreamOperation::Splice { .. })) {
                if let Some(first_segment) = segments.first() {
                    let input_file = stream.input_file.as_ref()
                        .ok_or_else(|| DriverError::Execution("No input file specified".to_string()))?;
                    
                    let poster_cmd = self.build_poster_command(input_file, &poster_path, first_segment.start)?;
                    self.execute_command(poster_cmd)?;
                    
                    println!("✅ Generated video: {}", sink.path);
                    println!("🖼️  Generated poster: {}", poster_path);
                }
            }
        } else {
            // For non-splice operations, just generate the video
            let cmd = self.build_ffmpeg_command(&stream, &sink.path)?;
            self.execute_command(cmd)?;
        }
        
        Ok(())
    }
} 