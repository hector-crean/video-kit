use std::ops::Range;
use std::fs::File;
use std::io::BufReader;
use gstreamer::ClockTime;
use serde::{Deserialize, Serialize};

use crate::{VideoEditCommand};
use crate::{init_macos_app, ExportFormat, LoopMode, ReverseMode, VideoEditor};
use tracing_subscriber;

#[derive(Serialize, Deserialize)]
pub struct Clip {
   pub edits: Vec<VideoEditCommand>,
   pub export: ExportFormat,
}


#[derive(Serialize, Deserialize)]
pub struct EditPipeline {
    input_uri: String,
    export_dir: String,
    clips: Vec<Clip>,
}

impl EditPipeline {
    pub fn from_json(json_path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(json_path)?;
        let reader = BufReader::new(file);
        let pipeline: EditPipeline = serde_json::from_reader(reader)?;
        Ok(pipeline)
    }

    // pub fn new(file_path: String, export_dir: String, clips: Vec<Clip>) -> Self {
    //     Self { input_uri: format!("file://{}", file_path), export_dir, clips }
    // }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {

    init_macos_app()?;
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("cutting=debug,info")
        .init();

    for clip in &self.clips {
        let mut editor = VideoEditor::new(self.input_uri.to_string());

        for command in &clip.edits {
            editor.add_command(command.clone());
        }
        editor.render_to_file(clip.export.clone(), &self.export_dir)?;
    }
    
    
    Ok(())
}

}



