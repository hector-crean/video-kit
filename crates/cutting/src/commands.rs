use gstreamer::{ClockTime, SeekFlags};
use serde::{Deserialize, Serialize};

/// Top-level video editing commands with file output support
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VideoEditCommand {
    /// Cut a section from the video (start to end)
    Cut {
        start: ClockTime,
        end: ClockTime,
    },
    
    /// Reverse video playback
    Reverse {
        flags: SeekFlags,
        rate: f64,
        mode: Option<ReverseMode>, // Added reverse mode
    },
    
    /// Loop video with different options
    Loop {
        mode: LoopMode,
        flags: SeekFlags,
    },
    
    /// Freeze video frame between start and end, while original audio continues playing
    Freeze {
        start: ClockTime,      // When to start the freeze
        end: ClockTime,        // When to end the freeze
        flags: SeekFlags,
        rate: f64,
    },
    
  
}

/// Different modes for reverse video playback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReverseMode {
    /// Reverse entire clip
    Full,
    /// Reverse specific time range
    Segment { start: ClockTime, end: ClockTime },
    /// Reverse with frame-level precision (slower but more accurate)
    FrameAccurate { fps: f64 },
}

impl ReverseMode {
    /// Create a segment reverse mode with millisecond precision
    pub fn segment_ms(start_ms: u64, end_ms: u64) -> Self {
        Self::Segment {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(end_ms),
        }
    }
    
    /// Create a segment reverse mode with second precision (kept for compatibility)
    pub fn segment(start_seconds: u64, end_seconds: u64) -> Self {
        Self::Segment {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(end_seconds),
        }
    }
    
    /// Create a frame-accurate reverse mode
    pub fn frame_accurate(fps: f64) -> Self {
        Self::FrameAccurate { fps }
    }
}

/// Different looping modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopMode {
    Infinite,
    Count(u32),
    Duration(ClockTime),
    UntilPosition(ClockTime),
    Segment { start: ClockTime, end: ClockTime },
}

impl LoopMode {
    /// Loop for a specific duration with millisecond precision
    pub fn duration_ms(duration_ms: u64) -> Self {
        Self::Duration(ClockTime::from_mseconds(duration_ms))
    }
    
    /// Loop for a specific duration (seconds - kept for compatibility)
    pub fn duration(duration_seconds: u64) -> Self {
        Self::Duration(ClockTime::from_seconds(duration_seconds))
    }
    
    /// Loop until reaching a specific position with millisecond precision
    pub fn until_position_ms(position_ms: u64) -> Self {
        Self::UntilPosition(ClockTime::from_mseconds(position_ms))
    }
    
    /// Loop until reaching a specific position (seconds - kept for compatibility) 
    pub fn until_position(position_seconds: u64) -> Self {
        Self::UntilPosition(ClockTime::from_seconds(position_seconds))
    }
    
    /// Loop a specific segment with millisecond precision
    pub fn segment_ms(start_ms: u64, end_ms: u64) -> Self {
        Self::Segment {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(end_ms),
        }
    }
    
    /// Loop a specific segment (seconds - kept for compatibility)
    pub fn segment(start_seconds: u64, end_seconds: u64) -> Self {
        Self::Segment {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(end_seconds),
        }
    }
}

// Convenience constructors for VideoEditCommand
impl VideoEditCommand {
    /// Cut video segment with millisecond precision using start and end times
    pub fn cut_range_ms(start_ms: u64, end_ms: u64) -> Self {
        Self::Cut {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(end_ms),
        }
    }
    
    /// Cut video segment with second precision using start and end times
    pub fn cut_range(start_seconds: u64, end_seconds: u64) -> Self {
        Self::Cut {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(end_seconds),
        }
    }
    
    /// Cut video segment with millisecond precision (legacy - uses duration)
    pub fn cut_ms(start_ms: u64, duration_ms: u64) -> Self {
        Self::Cut {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(start_ms + duration_ms),
        }
    }
    
    /// Cut video segment (legacy - uses duration, kept for compatibility)
    pub fn cut(start_seconds: u64, duration_seconds: u64) -> Self {
        Self::Cut {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(start_seconds + duration_seconds),
        }
    }
    
    /// Reverse video with default settings
    pub fn reverse() -> Self {
        Self::Reverse {
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: -1.0,
            mode: None, // Default to full reverse
        }
    }
    
    /// Reverse video with specific mode
    pub fn reverse_with_mode(mode: ReverseMode) -> Self {
        Self::Reverse {
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: -1.0,
            mode: Some(mode),
        }
    }
    
    /// Loop video a specific number of times
    pub fn loop_count(count: u32) -> Self {
        Self::Loop {
            mode: LoopMode::Count(count),
            flags: SeekFlags::FLUSH | SeekFlags::KEY_UNIT,
        }
    }

    /// Freeze frame between start and end times with millisecond precision
    pub fn freeze_range_ms(start_ms: u64, end_ms: u64) -> Self {
        Self::Freeze {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(end_ms),
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: 1.0, // Normal audio rate during freeze
        }
    }

    /// Freeze frame between start and end times with second precision
    pub fn freeze_range(start_seconds: u64, end_seconds: u64) -> Self {
        Self::Freeze {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(end_seconds),
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: 1.0, // Normal audio rate during freeze
        }
    }

    /// Freeze frame with millisecond precision (legacy - uses duration)
    pub fn freeze_ms(start_ms: u64, duration_ms: u64) -> Self {
        Self::Freeze {
            start: ClockTime::from_mseconds(start_ms),
            end: ClockTime::from_mseconds(start_ms + duration_ms),
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: 1.0, // Normal audio rate during freeze
        }
    }

    /// Freeze frame (legacy - uses duration, kept for compatibility)
    pub fn freeze(start_seconds: u64, duration_seconds: u64) -> Self {
        Self::Freeze {
            start: ClockTime::from_seconds(start_seconds),
            end: ClockTime::from_seconds(start_seconds + duration_seconds),
            flags: SeekFlags::FLUSH | SeekFlags::ACCURATE,
            rate: 1.0, // Normal audio rate during freeze
        }
    }
} 