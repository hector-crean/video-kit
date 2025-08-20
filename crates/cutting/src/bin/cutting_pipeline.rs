use cutting::edit_pipeline::EditPipeline;




fn main() -> Result<(), Box<dyn std::error::Error>> {

    let edit_pipeline = EditPipeline::from_json("/Users/hectorcrean/rust/video-kit/crates/cutting/input/input-videos.json".to_string())?;
    edit_pipeline.run()?;
    Ok(())
}