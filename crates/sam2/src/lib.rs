// crates/mask/src/python_integration.rs
use std::process::{Command, Stdio};
use std::io::Write;
use serde_json::{json, Value};
use tokio::process::Command as TokioCommand;

pub struct PythonMLProcessor {
    python_script_path: String,
    uv_env_path: Option<String>,
}

impl PythonMLProcessor {
    pub fn new(script_path: String, uv_env_path: Option<String>) -> Self {
        Self {
            python_script_path: script_path,
            uv_env_path,
        }
    }
    
    pub async fn process_image(&self, image_path: &str, parameters: Value) -> Result<Value, Box<dyn std::error::Error>> {
        let temp_input = tempfile::NamedTempFile::new()?;
        let temp_params = tempfile::NamedTempFile::new()?;
        
        // Write input data
        serde_json::to_writer(&temp_input, &json!({
            "image_path": image_path
        }))?;
        
        // Write parameters
        serde_json::to_writer(&temp_params, &parameters)?;
        
        // Build command
        let mut cmd = if let Some(uv_path) = &self.uv_env_path {
            let mut c = TokioCommand::new(uv_path);
            c.arg("run")
             .arg("python")
             .arg(&self.python_script_path);
            c
        } else {
            let mut c = TokioCommand::new("python");
            c.arg(&self.python_script_path);
            c
        };
        
        let output = cmd
            .arg("--input")
            .arg(temp_input.path())
            .arg("--params")
            .arg(temp_params.path())
            .output()
            .await?;
        
        if !output.status.success() {
            return Err(format!("Python script failed: {}", String::from_utf8_lossy(&output.stderr)).into());
        }
        
        let result: Value = serde_json::from_slice(&output.stdout)?;
        Ok(result)
    }
}