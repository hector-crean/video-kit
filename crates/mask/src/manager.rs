use std::{rc::Rc, sync::Arc};

use image::GrayImage;
use crate::{
    error::Result,
    types::ComputedOutline,
    pipeline::{Pipeline, builder::PipelineBuilder},
};
use serde::{Serialize, Deserialize};
use schemars::JsonSchema;
use strum::{Display, EnumString, EnumIter, IntoStaticStr, VariantNames};

#[derive(
    Debug, Clone, 
    Serialize, Deserialize, JsonSchema,
    Display, EnumString, EnumIter, VariantNames, IntoStaticStr,
    PartialEq
)]
#[serde(tag = "type", content = "params")]
#[strum(serialize_all = "snake_case")]
pub enum MaskManagerCommand {
    /// Extract basic outline from mask image
    #[serde(rename = "extract_outline")]
    ExtractOutline,
    
    /// Extract outline with Douglas-Peucker simplification
    #[serde(rename = "extract_outline_with_simplification")]
    ExtractOutlineWithSimplification { 
        #[schemars(range(min = 0.1, max = 10.0))]
        tolerance: f32 
    },
    
    /// Extract outline with hole detection for complex shapes
    #[serde(rename = "extract_outline_with_hole_detection")]
    ExtractOutlineWithHoleDetection,
    
    /// Extract outline using a named custom extractor
    #[serde(rename = "extract_outline_with_custom_extractor")]
    ExtractOutlineWithCustomExtractor { 
        #[schemars(length(min = 1, max = 50))]
        extractor_name: String 
    },
}

impl MaskManagerCommand {
    /// Get the JSON schema for all commands
    pub fn schema() -> schemars::schema::RootSchema {
        schemars::schema_for!(MaskManagerCommand)
    }
    
    /// Get a list of all available command names
    pub fn command_names() -> &'static [&'static str] {
        <Self as VariantNames>::VARIANTS
    }
    
    /// Get a description of the command
    pub fn description(&self) -> &'static str {
        match self {
            Self::ExtractOutline => "Extract basic polyline outline from mask image",
            Self::ExtractOutlineWithSimplification { .. } => "Extract outline with Douglas-Peucker simplification to reduce point count",
            Self::ExtractOutlineWithHoleDetection => "Extract outline with hole detection for complex shapes like donuts",
            Self::ExtractOutlineWithCustomExtractor { .. } => "Extract outline using a named custom extraction algorithm",
        }
    }
    
    /// Get parameter requirements for the command
    pub fn parameters_info(&self) -> Vec<(&'static str, &'static str, bool)> {
        match self {
            Self::ExtractOutline => vec![],
            Self::ExtractOutlineWithSimplification { .. } => vec![
                ("tolerance", "Simplification tolerance (0.1-10.0, higher = more simplified)", true)
            ],
            Self::ExtractOutlineWithHoleDetection => vec![],
            Self::ExtractOutlineWithCustomExtractor { .. } => vec![
                ("extractor_name", "Name of the custom extractor to use", true)
            ],
        }
    }
}

/// Legacy MaskManager for backward compatibility
#[derive(Clone)]
pub struct MaskManager {
    image: Option<GrayImage>,
    pipeline: Arc<Pipeline>,
}

impl MaskManager {
    pub fn new() -> Self {
        Self {
            image: None,
            pipeline: Arc::new(PipelineBuilder::build_simple(128)),
        }
    }

    /// Create a new MaskManager with a custom pipeline
    pub fn with_pipeline(pipeline: Pipeline) -> Self {
        Self {
            image: None,
            pipeline: Arc::new(pipeline),
        }
    }
    
    /// Load a mask image from file
    pub fn load_image(&mut self, path: &str) -> Result<()> {
        let img = image::open(path)?;
        self.image = Some(img.to_luma8());
        Ok(())
    }
    
    /// Load a mask image from memory
    pub fn load_image_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let img = image::load_from_memory(bytes)?;
        self.image = Some(img.to_luma8());
        Ok(())
    }
    
    /// Set the mask image directly
    pub fn set_image(&mut self, image: GrayImage) {
        self.image = Some(image);
    }

    pub fn execute(&self, command: MaskManagerCommand) -> Result<ComputedOutline> {
        let image = self.image.as_ref()
            .ok_or(crate::error::MaskError::NoImageLoaded)?;
            
        match command {
            MaskManagerCommand::ExtractOutline => {
                self.pipeline.process(image)
            }
            MaskManagerCommand::ExtractOutlineWithSimplification { tolerance } => {
                let pipeline = PipelineBuilder::build_with_simplification(128, tolerance);
                pipeline.process(image)
            }
            MaskManagerCommand::ExtractOutlineWithHoleDetection => {
                let pipeline = PipelineBuilder::build_with_holes(128);
                pipeline.process(image)
            }
            MaskManagerCommand::ExtractOutlineWithCustomExtractor { extractor_name: _ } => {
                // For backwards compatibility, just use the current pipeline
                self.pipeline.process(image)
            }
        }
    }
}

impl Default for MaskManager {
    fn default() -> Self {
        Self::new()
    }
}