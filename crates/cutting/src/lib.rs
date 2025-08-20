pub mod edit_pipeline;
pub mod error;
pub mod commands;
pub mod formats;
pub mod editor;

// Import common dependencies for re-export
use gstreamer::{self as gst, prelude::*, ClockTime, SeekFlags, SeekType, glib};
use gstreamer_app as gst_app;
use gstreamer_editing_services::{self as ges, prelude::*};
use gstreamer_pbutils::{EncodingAudioProfile, EncodingVideoProfile, EncodingContainerProfile};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, path::Path, str::FromStr};
use thiserror::Error;
use tracing::{info, debug, warn, error, trace};

// macOS support module
pub mod macos;
pub use macos::{init_macos_app, init_macos_background_app};

// Re-export all public types from modules
pub use error::{VideoEditError, GStreamerResultExt};
pub use commands::{VideoEditCommand, ReverseMode, LoopMode};
pub use formats::{ExportFormat, VideoFormat, ImageFormat};
pub use editor::VideoEditor;