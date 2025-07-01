use deepgram::{
    common::{
        audio_source::AudioSource,
        batch_response::{Response, Utterance},
        options::{Language, Model, Options},
    },
    Deepgram, DeepgramError,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
use tempfile::NamedTempFile;

// Re-export Deepgram types for convenience
pub use deepgram::common::batch_response::{Word as DeepgramWord, Utterance as SpeechSegment};

#[derive(Error, Debug)]
pub enum SubtitleError {
    #[error("Deepgram API error: {0}")]
    Deepgram(#[from] DeepgramError),
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),
    #[error("No transcription results available")]
    NoResults,
    #[error("FFmpeg error: {0}")]
    FFmpegError(String),
    #[error("Audio extraction failed: {0}")]
    AudioExtractionFailed(String),
}

/// Represents a single subtitle/caption with timing information
/// This is a simplified view of Deepgram's Utterance for subtitle export
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Subtitle {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds  
    pub end: f64,
    /// The spoken text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Speaker identification (if available)
    pub speaker: Option<String>,
    /// Language detected/used
    pub language: Option<String>,
}

impl From<&Utterance> for Subtitle {
    fn from(utterance: &Utterance) -> Self {
        Self {
            start: utterance.start,
            end: utterance.end,
            text: utterance.transcript.clone(),
            confidence: utterance.confidence,
            speaker: utterance.speaker.map(|s| format!("Speaker {}", s)),
            language: None, // Will be set by the extractor
        }
    }
}

/// Configuration for subtitle extraction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SubtitleConfig {
    /// Language for transcription (auto-detect if None)
    pub language: Option<String>,
    /// Whether to include speaker identification (diarization)
    pub identify_speakers: bool,
    /// Whether to include word-level timestamps
    pub word_timestamps: bool,
    /// Whether to filter out profanity
    pub filter_profanity: bool,
    /// Custom vocabulary or phrases to improve accuracy
    pub custom_vocabulary: Vec<String>,
    /// Model selection (e.g., "nova-2", "enhanced", "base")
    pub model: Option<String>,
    /// Whether to enable punctuation
    pub punctuation: bool,
    /// Whether to enable utterances (sentence-level segmentation)
    pub utterances: bool,
    /// Audio quality for extracted audio (128, 192, 320 kbps)
    pub audio_bitrate: u32,
    /// Whether to extract audio from video files automatically
    pub auto_extract_audio: bool,
}

impl Default for SubtitleConfig {
    fn default() -> Self {
        Self {
            language: None, // Auto-detect
            identify_speakers: false,
            word_timestamps: true,
            filter_profanity: false,
            custom_vocabulary: Vec::new(),
            model: Some("nova-3".to_string()),
            punctuation: true,
            utterances: true,
            audio_bitrate: 128, // 128 kbps for good quality/size balance
            auto_extract_audio: true,
        }
    }
}

/// Main client for subtitle extraction using Deepgram
pub struct SubtitleExtractor {
    deepgram: Deepgram,
}

impl SubtitleExtractor {
    /// Create a new subtitle extractor with Deepgram API key
    pub fn new(api_key: String) -> Result<Self, SubtitleError> {
        let deepgram = Deepgram::new(&api_key)?;
        Ok(Self { deepgram })
    }

    /// Extract subtitles from audio/video file
    pub async fn extract_subtitles(
        &self,
        file_path: &Path,
        config: &SubtitleConfig,
    ) -> Result<Vec<Subtitle>, SubtitleError> {
        let utterances = self.extract_speech_segments(file_path, config).await?;
        Ok(utterances.iter().map(|utterance| {
            let mut subtitle = Subtitle::from(utterance);
            subtitle.language = config.language.clone();
            subtitle
        }).collect())
    }

    /// Extract detailed speech segments with word-level timing (returns Deepgram Utterances)
    pub async fn extract_speech_segments(
        &self,
        file_path: &Path,
        config: &SubtitleConfig,
    ) -> Result<Vec<Utterance>, SubtitleError> {
        // Determine if we need to extract audio from video
        let audio_path = if self.is_video_file(file_path) && config.auto_extract_audio {
            self.extract_audio_from_video(file_path, config).await?
        } else {
            AudioFile::Existing(file_path.to_path_buf())
        };

        let result = self.process_audio_file(&audio_path.path(), config).await;

        // Clean up temporary file if created
        if let AudioFile::Temporary(_) = audio_path {
            // File will be automatically cleaned up when NamedTempFile is dropped
        }

        result
    }

    /// Process an audio file with Deepgram
    async fn process_audio_file(
        &self,
        audio_path: &Path,
        config: &SubtitleConfig,
    ) -> Result<Vec<Utterance>, SubtitleError> {
        // Read the audio file
        let file = tokio::fs::File::open(audio_path).await?;
        let audio_source = AudioSource::from_buffer_with_mime_type(
            file,
            Self::get_mime_type(audio_path)?,
        );

        // Build transcription options
        let mut options_builder = Options::builder()
            .punctuate(config.punctuation)
            .utterances(config.utterances);

        if config.identify_speakers {
            options_builder = options_builder.diarize(true);
        }

        if let Some(ref language) = config.language {
            let lang = match language.as_str() {
                "en" => Language::en_US,
                "es" => Language::es,
                "fr" => Language::fr,
                "de" => Language::de,
                "it" => Language::it,
                "pt" => Language::pt,
                "ru" => Language::ru,
                "ja" => Language::ja,
                "ko" => Language::ko,
                "zh" => Language::zh,
                "nl" => Language::nl,
                "tr" => Language::tr,
                "pl" => Language::pl,
                "sv" => Language::sv,
                "bg" => Language::bg,
                "cs" => Language::cs,
                "da" => Language::da,
                "el" => Language::el,
                "fi" => Language::fi,
                "et" => Language::et,
                _ => Language::en_US, // Default to English
            };
            options_builder = options_builder.language(lang);
        }

        if config.filter_profanity {
            options_builder = options_builder.profanity_filter(true);
        }

        if let Some(ref model) = config.model {
            let model_enum = match model.as_str() {
                "nova-3" => Model::Nova3,
                "nova-2" => Model::Nova2,
                "nova" => Model::Nova3, // Updated to Nova3 (recommended)
                "enhanced" => Model::Nova3, // Updated to Nova3 (recommended)
                "base" => Model::Nova3, // Updated to Nova3 (recommended)
                _ => Model::Nova2, // Default to nova-2
            };
            options_builder = options_builder.model(model_enum);
        }

        // Add custom vocabulary if provided
        if !config.custom_vocabulary.is_empty() {
            let keyword_refs: Vec<&str> = config.custom_vocabulary.iter().map(|s| s.as_str()).collect();
            options_builder = options_builder.keywords(keyword_refs);
        }

        let options = options_builder.build();

        // Execute transcription
        let response = self.deepgram
            .transcription()
            .prerecorded(audio_source, &options)
            .await?;

        // Parse the response into our format
        self.extract_utterances_from_response(response)
    }

    /// Extract audio from video file to temporary MP3
    async fn extract_audio_from_video(
        &self,
        video_path: &Path,
        config: &SubtitleConfig,
    ) -> Result<AudioFile, SubtitleError> {
        println!("🎵 Extracting audio from video: {:?}", video_path);

        // Create a temporary file with .mp3 extension
        let temp_file = NamedTempFile::with_suffix(".mp3")
            .map_err(|e| SubtitleError::AudioExtractionFailed(format!("Failed to create temp file: {}", e)))?;
        
        let temp_path = temp_file.path();

        // Use FFmpeg to extract audio with explicit path and library environment
        let output = Command::new("/opt/homebrew/bin/ffmpeg")
            .env("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
            .args([
                "-i", video_path.to_str().unwrap(),
                "-vn", // No video
                "-acodec", "mp3",
                "-ab", &format!("{}k", config.audio_bitrate), // Audio bitrate
                "-ac", "2", // Stereo
                "-ar", "44100", // Sample rate
                "-y", // Overwrite output file
                temp_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| SubtitleError::FFmpegError(format!("Failed to execute ffmpeg: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SubtitleError::AudioExtractionFailed(format!(
                "FFmpeg failed: {}", stderr
            )));
        }

        println!("✅ Audio extracted to temporary file ({} kbps)", config.audio_bitrate);

        Ok(AudioFile::Temporary(temp_file))
    }

    /// Check if file is a video format that needs audio extraction
    fn is_video_file(&self, file_path: &Path) -> bool {
        if let Some(extension) = file_path.extension().and_then(|ext| ext.to_str()) {
            matches!(extension.to_lowercase().as_str(),
                "mp4" | "avi" | "mov" | "webm" | "mkv" | "wmv" | "m4v" | "3gp" | "flv"
            )
        } else {
            false
        }
    }

    /// Check if the file format is supported for direct audio processing
    pub fn supports_format(&self, extension: &str) -> bool {
        matches!(extension.to_lowercase().as_str(), 
            "mp3" | "mp4" | "wav" | "flac" | "ogg" | "m4a" | "webm" | "avi" | "mov" | "aac" |
            "mkv" | "wmv" | "m4v" | "3gp" | "flv"
        )
    }

    fn get_mime_type(file_path: &Path) -> Result<&'static str, SubtitleError> {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| SubtitleError::InvalidFormat("No file extension".to_string()))?
            .to_lowercase();

        let mime_type = match extension.as_str() {
            "mp3" => "audio/mpeg",
            "mp4" => "video/mp4",
            "wav" => "audio/wav",
            "flac" => "audio/flac",
            "ogg" => "audio/ogg",
            "m4a" => "audio/mp4",
            "webm" => "video/webm",
            "avi" => "video/x-msvideo",
            "mov" => "video/quicktime",
            "aac" => "audio/aac",
            _ => return Err(SubtitleError::InvalidFormat(format!("Unsupported format: {}", extension))),
        };

        Ok(mime_type)
    }

    fn extract_utterances_from_response(
        &self,
        response: Response,
    ) -> Result<Vec<Utterance>, SubtitleError> {
        // Use utterances if available (preferred for sentence-level segments with timing)
        if let Some(utterances) = response.results.utterances {
            if !utterances.is_empty() {
                return Ok(utterances);
            }
        }

        // If no utterances are available, we need utterances mode enabled
        // This ensures we get proper sentence-level segmentation with timing
        Err(SubtitleError::NoResults)
    }
}

/// Represents either an existing audio file or a temporary extracted audio file
enum AudioFile {
    Existing(PathBuf),
    Temporary(NamedTempFile),
}

impl AudioFile {
    fn path(&self) -> &Path {
        match self {
            AudioFile::Existing(path) => path,
            AudioFile::Temporary(temp_file) => temp_file.path(),
        }
    }
}

/// Smart video clipping utilities based on speech analysis
pub struct SmartClipper {
    extractor: SubtitleExtractor,
}

impl SmartClipper {
    /// Create a new smart clipper with Deepgram API key
    pub fn new(api_key: String) -> Result<Self, SubtitleError> {
        Ok(Self {
            extractor: SubtitleExtractor::new(api_key)?,
        })
    }

    /// Find optimal cut points that avoid interrupting speech
    pub async fn find_optimal_cuts(
        &self,
        video_path: &Path,
        desired_segments: &[(f64, f64)], // (start, end) pairs
        config: &SubtitleConfig,
    ) -> Result<Vec<OptimalCut>, SubtitleError> {
        let utterances = self.extractor.extract_speech_segments(video_path, config).await?;
        let mut optimal_cuts = Vec::new();

        for &(desired_start, desired_end) in desired_segments {
            let optimal_cut = self.find_best_cut_points(
                desired_start,
                desired_end,
                &utterances,
            );
            optimal_cuts.push(optimal_cut);
        }

        Ok(optimal_cuts)
    }

    /// Analyze speech density to find natural break points
    pub async fn find_natural_breaks(
        &self,
        video_path: &Path,
        min_break_duration: f64, // minimum silence duration to consider a break
        config: &SubtitleConfig,
    ) -> Result<Vec<f64>, SubtitleError> {
        let utterances = self.extractor.extract_speech_segments(video_path, config).await?;
        let mut breaks = Vec::new();

        for i in 0..utterances.len().saturating_sub(1) {
            let current_end = utterances[i].end;
            let next_start = utterances[i + 1].start;
            let gap_duration = next_start - current_end;

            if gap_duration >= min_break_duration {
                // Find the midpoint of the silence
                breaks.push(current_end + gap_duration / 2.0);
            }
        }

        Ok(breaks)
    }

    fn find_best_cut_points(
        &self,
        desired_start: f64,
        desired_end: f64,
        utterances: &[Utterance],
    ) -> OptimalCut {
        let mut best_start = desired_start;
        let mut best_end = desired_end;
        let mut start_confidence = 1.0;
        let mut end_confidence = 1.0;
        let mut adjustment_reason = "No adjustment needed".to_string();

        // Check if desired start is in the middle of speech
        for utterance in utterances {
            if utterance.start <= desired_start && utterance.end >= desired_start {
                // Desired start cuts through speech - find better position
                let distance_to_start = desired_start - utterance.start;
                let distance_to_end = utterance.end - desired_start;
                
                if distance_to_start < distance_to_end {
                    // Closer to utterance start, move cut to before utterance
                    best_start = utterance.start - 0.1; // Small buffer
                    start_confidence = 0.9;
                } else {
                    // Closer to utterance end, move cut to after utterance
                    best_start = utterance.end + 0.1; // Small buffer
                    start_confidence = 0.9;
                }
                adjustment_reason = "Adjusted start to avoid cutting mid-speech".to_string();
                break;
            }
        }

        // Check if desired end is in the middle of speech
        for utterance in utterances {
            if utterance.start <= desired_end && utterance.end >= desired_end {
                // Desired end cuts through speech - find better position
                let distance_to_start = desired_end - utterance.start;
                let distance_to_end = utterance.end - desired_end;
                
                if distance_to_start < distance_to_end {
                    // Closer to utterance start, move cut to before utterance
                    best_end = utterance.start - 0.1; // Small buffer
                    end_confidence = 0.9;
                } else {
                    // Closer to utterance end, move cut to after utterance
                    best_end = utterance.end + 0.1; // Small buffer
                    end_confidence = 0.8;
                }
                
                if adjustment_reason == "No adjustment needed" {
                    adjustment_reason = "Adjusted end to avoid cutting mid-speech".to_string();
                } else {
                    adjustment_reason = "Adjusted both start and end to avoid cutting mid-speech".to_string();
                }
                break;
            }
        }

        OptimalCut {
            original_start: desired_start,
            original_end: desired_end,
            optimal_start: best_start,
            optimal_end: best_end,
            start_confidence,
            end_confidence,
            adjustment_reason,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimalCut {
    pub original_start: f64,
    pub original_end: f64,
    pub optimal_start: f64,
    pub optimal_end: f64,
    pub start_confidence: f64,
    pub end_confidence: f64,
    pub adjustment_reason: String,
}

/// Export subtitles to various formats
pub struct SubtitleExporter;

impl SubtitleExporter {
    /// Export to SRT format
    pub fn to_srt(subtitles: &[Subtitle]) -> String {
        let mut output = String::new();
        
        for (i, subtitle) in subtitles.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str(&format!(
                "{} --> {}\n",
                Self::format_timestamp_srt(subtitle.start),
                Self::format_timestamp_srt(subtitle.end)
            ));
            output.push_str(&format!("{}\n\n", subtitle.text));
        }
        
        output
    }
    
    /// Export to VTT format
    pub fn to_vtt(subtitles: &[Subtitle]) -> String {
        let mut output = String::from("WEBVTT\n\n");
        
        for subtitle in subtitles {
            output.push_str(&format!(
                "{} --> {}\n",
                Self::format_timestamp_vtt(subtitle.start),
                Self::format_timestamp_vtt(subtitle.end)
            ));
            output.push_str(&format!("{}\n\n", subtitle.text));
        }
        
        output
    }
    
    /// Export to JSON format
    pub fn to_json(subtitles: &[Subtitle]) -> Result<String, SubtitleError> {
        Ok(serde_json::to_string_pretty(subtitles)?)
    }

    /// Export Deepgram utterances directly to SRT format
    pub fn utterances_to_srt(utterances: &[Utterance]) -> String {
        let mut output = String::new();
        
        for (i, utterance) in utterances.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str(&format!(
                "{} --> {}\n",
                Self::format_timestamp_srt(utterance.start),
                Self::format_timestamp_srt(utterance.end)
            ));
            output.push_str(&format!("{}\n\n", utterance.transcript));
        }
        
        output
    }

    /// Export Deepgram utterances directly to VTT format
    pub fn utterances_to_vtt(utterances: &[Utterance]) -> String {
        let mut output = String::from("WEBVTT\n\n");
        
        for utterance in utterances {
            output.push_str(&format!(
                "{} --> {}\n",
                Self::format_timestamp_vtt(utterance.start),
                Self::format_timestamp_vtt(utterance.end)
            ));
            output.push_str(&format!("{}\n\n", utterance.transcript));
        }
        
        output
    }
    
    fn format_timestamp_srt(seconds: f64) -> String {
        let total_ms = (seconds * 1000.0) as u64;
        let ms = total_ms % 1000;
        let total_seconds = total_ms / 1000;
        let secs = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mins = total_minutes % 60;
        let hours = total_minutes / 60;
        
        format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
    }
    
    fn format_timestamp_vtt(seconds: f64) -> String {
        let total_ms = (seconds * 1000.0) as u64;
        let ms = total_ms % 1000;
        let total_seconds = total_ms / 1000;
        let secs = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mins = total_minutes % 60;
        let hours = total_minutes / 60;
        
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
    }
}

/// Convenience function for extracting subtitles with Deepgram
pub async fn extract_subtitles(
    api_key: String,
    video_path: &Path,
    config: Option<SubtitleConfig>,
) -> Result<Vec<Subtitle>, SubtitleError> {
    let extractor = SubtitleExtractor::new(api_key)?;
    let config = config.unwrap_or_default();
    extractor.extract_subtitles(video_path, &config).await
}

/// Convenience function for extracting speech segments (Deepgram Utterances) with Deepgram
pub async fn extract_speech_segments(
    api_key: String,
    video_path: &Path,
    config: Option<SubtitleConfig>,
) -> Result<Vec<Utterance>, SubtitleError> {
    let extractor = SubtitleExtractor::new(api_key)?;
    let config = config.unwrap_or_default();
    extractor.extract_speech_segments(video_path, &config).await
}

/// Convenience function for finding optimal cuts with Deepgram
pub async fn find_optimal_cuts(
    api_key: String,
    video_path: &Path,
    desired_segments: &[(f64, f64)],
    config: Option<SubtitleConfig>,
) -> Result<Vec<OptimalCut>, SubtitleError> {
    let clipper = SmartClipper::new(api_key)?;
    let config = config.unwrap_or_default();
    clipper.find_optimal_cuts(video_path, desired_segments, &config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_subtitle_creation() {
        let subtitle = Subtitle {
            start: 0.0,
            end: 5.0,
            text: "Hello, world!".to_string(),
            confidence: 0.95,
            speaker: Some("Speaker 1".to_string()),
            language: Some("en".to_string()),
        };
        
        assert_eq!(subtitle.start, 0.0);
        assert_eq!(subtitle.end, 5.0);
        assert_eq!(subtitle.text, "Hello, world!");
    }
    
    #[test]
    fn test_srt_export() {
        let subtitles = vec![
            Subtitle {
                start: 0.0,
                end: 2.5,
                text: "Hello, world!".to_string(),
                confidence: 0.95,
                speaker: None,
                language: Some("en".to_string()),
            },
            Subtitle {
                start: 3.0,
                end: 6.0,
                text: "How are you today?".to_string(),
                confidence: 0.92,
                speaker: None,
                language: Some("en".to_string()),
            },
        ];
        
        let srt = SubtitleExporter::to_srt(&subtitles);
        assert!(srt.contains("Hello, world!"));
        assert!(srt.contains("How are you today?"));
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
    }

    #[test]
    fn test_format_support() {
        let extractor = SubtitleExtractor::new("test_key".to_string()).unwrap();
        assert!(extractor.supports_format("mp4"));
        assert!(extractor.supports_format("wav"));
        assert!(extractor.supports_format("MP3"));
        assert!(extractor.supports_format("mkv"));
        assert!(!extractor.supports_format("txt"));
    }

    #[test]
    fn test_video_detection() {
        let extractor = SubtitleExtractor::new("test_key".to_string()).unwrap();
        assert!(extractor.is_video_file(Path::new("test.mp4")));
        assert!(extractor.is_video_file(Path::new("test.avi")));
        assert!(extractor.is_video_file(Path::new("test.mov")));
        assert!(!extractor.is_video_file(Path::new("test.mp3")));
        assert!(!extractor.is_video_file(Path::new("test.wav")));
    }
}
