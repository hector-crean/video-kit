//cargo run --package mask --bin mask_mcp_server 
use mask::mcp::MaskMcpServer;
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::{util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    // Set up logging to stderr (MCP uses stdout for protocol communication)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting Mask Outline Extraction MCP server");

    // Create the MCP server
    let server = MaskMcpServer::new();
    
    // Serve over stdio transport
    let service = match server.serve(stdio()).await {
        Ok(service) => service,
        Err(e) => {
            tracing::error!("Failed to start MCP server: {:?}", e);
            return Err(e.into());
        }
    };

    tracing::info!("MCP server started, listening on stdio");

    // Handle graceful shutdown
    tokio::select! {
        result = service.waiting() => {
            match result {
                Ok(_) => tracing::info!("MCP server completed successfully"),
                Err(e) => {
                    tracing::error!("MCP server error: {:?}", e);
                    return Err(e.into());
                }
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received Ctrl+C, shutting down gracefully");
        }
    }

    tracing::info!("MCP server shut down");
    Ok(())
}