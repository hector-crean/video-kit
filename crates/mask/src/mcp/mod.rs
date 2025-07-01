use crate::{
    manager::{MaskManager, MaskManagerCommand}, 
    typed_geojson::{TypedGeoJson, MaskGeoJson}, 
    types::ComputedOutline
};
use rmcp::{
    handler::server::tool::IntoCallToolResult, model::{CallToolResult, Content, ErrorCode, ServerCapabilities, ServerInfo}, schemars, tool, Error as McpError, ServerHandler
};
use serde::{Deserialize, Serialize};
use ts_rs::TS;
use std::sync::{Arc, RwLock};

/// Request for loading an image from base64 data
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LoadImageRequest {
    #[schemars(description = "Path to the image file")]
    pub path: String,
}

/// Request for simplification with tolerance parameter
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SimplificationRequest {
    #[schemars(
        description = "Simplification tolerance (higher = more simplified)",
        range(min = 0.1, max = 10.0)
    )]
    pub tolerance: f32,
}

/// Request for custom extractor
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CustomExtractorRequest {
    #[schemars(
        description = "Name of the custom extractor to use",
        length(min = 1, max = 50)
    )]
    pub extractor_name: String,
}

/// Response containing outline extraction results
#[derive(Debug, Serialize, schemars::JsonSchema, TS)]
#[ts(export)]
pub struct OutlineResponse {
    #[schemars(description = "Number of shapes extracted")]
    pub shapes_count: usize,
    #[schemars(description = "Original image dimensions")]
    pub image_dimensions: ImageDimensions,
    #[schemars(description = "GeoJSON representation of the extracted shapes")]
    pub geojson: serde_json::Value,
}

#[derive(Debug, Serialize, schemars::JsonSchema, TS)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

/// MCP Server for mask outline extraction
#[derive(Clone)]
pub struct MaskMcpServer {
    manager: Arc<RwLock<Option<MaskManager>>>,
}

impl MaskMcpServer {
    pub fn new() -> Self {
        Self { manager: Arc::new(RwLock::new(None)) }
    }

    /// Convert ComputedOutline to OutlineResponse
    fn result_to_response(&self, result: ComputedOutline) -> Result<OutlineResponse, String> {
        let typed_geojson = result.to_typed_geojson()
            .map_err(|e| format!("Failed to convert to typed GeoJSON: {}", e))?;
            
        Ok(OutlineResponse {
            shapes_count: result.shapes.len(),
            image_dimensions: ImageDimensions {
                width: result.image_width,
                height: result.image_height,
            },
            geojson: serde_json::to_value(typed_geojson).unwrap(),
        })
    }
}

impl Default for MaskMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoCallToolResult for OutlineResponse {
    fn into_call_tool_result(self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&self).unwrap_or_else(|_| format!("{:?}", self)),
        )]))
    }
}

// Add error response helper
impl OutlineResponse {
    fn error(message: String) -> Self {
        Self {
            shapes_count: 0,
            image_dimensions: ImageDimensions { width: 0, height: 0 },
            geojson: serde_json::json!({ "error": message }),
        }
    }
}

#[tool(tool_box)]
impl MaskMcpServer {
    #[tool(description = "Load a mask image from file path")]
    fn load_image(&self, #[tool(aggr)] LoadImageRequest { path }: LoadImageRequest) -> String {
        let mut manager = MaskManager::new();
        if let Err(e) = manager.load_image(&path) {
            return format!("Failed to load image from {}: {}", path, e);
        }
            
        *self.manager.write().unwrap() = Some(manager);
        format!("Image loaded successfully from {}", path)
    }

    #[tool(description = "Extract basic outline from the loaded mask image")]
    fn extract_outline(&self) -> OutlineResponse {
        let manager_ref = self.manager.read().unwrap();
        let manager = match manager_ref.as_ref() {
            Some(manager) => manager,
            None => return OutlineResponse::error("No image loaded. Please load an image first.".to_string()),
        };
            
        let result = match manager.execute(MaskManagerCommand::ExtractOutline) {
            Ok(result) => result,
            Err(e) => return OutlineResponse::error(format!("Outline extraction failed: {}", e)),
        };
            
        match self.result_to_response(result) {
            Ok(response) => response,
            Err(e) => OutlineResponse::error(e),
        }
    }

    #[tool(description = "Extract outline with Douglas-Peucker simplification to reduce point count")]
    fn extract_outline_with_simplification(
        &self, 
        #[tool(aggr)] SimplificationRequest { tolerance }: SimplificationRequest
    ) -> OutlineResponse {
        let manager_ref = self.manager.read().unwrap();
        let manager = match manager_ref.as_ref() {
            Some(manager) => manager,
            None => return OutlineResponse::error("No image loaded. Please load an image first.".to_string()),
        };
            
        let result = match manager.execute(MaskManagerCommand::ExtractOutlineWithSimplification { tolerance }) {
            Ok(result) => result,
            Err(e) => return OutlineResponse::error(format!("Simplified outline extraction failed: {}", e)),
        };
            
        match self.result_to_response(result) {
            Ok(response) => response,
            Err(e) => OutlineResponse::error(e),
        }
    }

    #[tool(description = "Extract outline with hole detection for complex shapes like donuts")]
    fn extract_outline_with_holes(&self) -> OutlineResponse {
        let manager_ref = self.manager.read().unwrap();
        let manager = match manager_ref.as_ref() {
            Some(manager) => manager,
            None => return OutlineResponse::error("No image loaded. Please load an image first.".to_string()),
        };
            
        let result = match manager.execute(MaskManagerCommand::ExtractOutlineWithHoleDetection) {
            Ok(result) => result,
            Err(e) => return OutlineResponse::error(format!("Hole-aware outline extraction failed: {}", e)),
        };
            
        match self.result_to_response(result) {
            Ok(response) => response,
            Err(e) => OutlineResponse::error(e),
        }
    }

    #[tool(description = "Extract outline using a named custom extraction algorithm")]
    fn extract_outline_with_custom_extractor(
        &self,
        #[tool(aggr)] CustomExtractorRequest { extractor_name }: CustomExtractorRequest
    ) -> OutlineResponse {
        let manager_ref = self.manager.read().unwrap();
        let manager = match manager_ref.as_ref() {
            Some(manager) => manager,
            None => return OutlineResponse::error("No image loaded. Please load an image first.".to_string()),
        };
            
        let result = match manager.execute(MaskManagerCommand::ExtractOutlineWithCustomExtractor { extractor_name }) {
            Ok(result) => result,
            Err(e) => return OutlineResponse::error(format!("Custom extraction failed: {}", e)),
        };
            
        match self.result_to_response(result) {
            Ok(response) => response,
            Err(e) => OutlineResponse::error(e),
        }
    }

    #[tool(description = "Get information about available commands and their parameters")]
    fn get_command_info(&self) -> String {
        let mut info = String::new();
        info.push_str("Available MaskManagerCommands:\n\n");
        
        for (i, name) in MaskManagerCommand::command_names().iter().enumerate() {
            info.push_str(&format!("{}. {}\n", i + 1, name));
        }
        
        info.push_str("\nCommand Details:\n");
        let commands = vec![
            MaskManagerCommand::ExtractOutline,
            MaskManagerCommand::ExtractOutlineWithSimplification { tolerance: 2.0 },
            MaskManagerCommand::ExtractOutlineWithHoleDetection,
            MaskManagerCommand::ExtractOutlineWithCustomExtractor { 
                extractor_name: "example".to_string() 
            },
        ];
        
        for cmd in commands {
            info.push_str(&format!("\n• {}\n", cmd));
            info.push_str(&format!("  Description: {}\n", cmd.description()));
            
            let params = cmd.parameters_info();
            if !params.is_empty() {
                info.push_str("  Parameters:\n");
                for (name, desc, required) in params {
                    let req_marker = if required { " (required)" } else { " (optional)" };
                    info.push_str(&format!("    - {}{}: {}\n", name, req_marker, desc));
                }
            }
        }
        
        info
    }

    #[tool(description = "Get the JSON schema for MaskManagerCommand")]
    fn get_command_schema(&self) -> String {
        let schema = MaskManagerCommand::schema();
        serde_json::to_string_pretty(&schema)
            .unwrap_or_else(|e| format!("Failed to serialize schema: {}", e))
    }
}

#[tool(tool_box)]
impl ServerHandler for MaskMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Mask Outline Extraction Server - Extract polyline outlines from mask images with support for complex shapes, holes, and advanced geometric algorithms from the geo crate.".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

