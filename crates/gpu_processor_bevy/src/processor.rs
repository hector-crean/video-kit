//! crates/gpu_processor/src/processor.rs

use crate::image_format::GpuImageFormat;
use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePass, ComputePipelineDescriptor, PipelineCache,
            ShaderStages, StorageTextureAccess, TextureSampleType, TextureViewDimension,
        },
        renderer::RenderDevice,
        texture::GpuImage,
    },
};
use std::borrow::Cow;

/// A handle to a compute shader for image processing.
#[derive(Asset, TypePath, Clone)]
pub struct GpuImageProcessor {
    pub shader: Handle<Shader>,
}

/// State for a single GPU image processing task.
#[derive(Resource)]
pub struct ImageProcessorState {
    pipeline: CachedComputePipelineId,
    bind_group: Option<BindGroup>,
}

impl GpuImageProcessor {
    /// Creates a new image processor with the given compute shader.
    pub fn new(shader: Handle<Shader>) -> Self {
        Self { shader }
    }

    /// Prepares the processing pipeline and bind group.
    pub fn prepare(
        &self,
        render_device: &RenderDevice,
        pipeline_cache: &mut PipelineCache,
        input_image: &GpuImage,
        output_image: &GpuImage,
    ) -> ImageProcessorState {
        let bind_group_layout =
            self.create_bind_group_layout(render_device, output_image.texture_format.into());

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("image_processing_pipeline")),
            layout: vec![bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("main"),
            push_constant_ranges: Vec::new(),
        });

        let bind_group =
            self.create_bind_group(render_device, &bind_group_layout, input_image, output_image);

        ImageProcessorState {
            pipeline,
            bind_group: Some(bind_group),
        }
    }

    /// Dispatches the compute shader to process the image.
    pub fn process<'a>(
        &self,
        pass: &mut ComputePass<'a>,
        state: &'a ImageProcessorState,
        image_dimensions: (u32, u32),
        pipeline_cache: &'a PipelineCache,
    ) {
        let (width, height) = image_dimensions;
        let workgroup_size = (8, 8, 1); // Standard workgroup size

        let pipeline = match pipeline_cache.get_compute_pipeline(state.pipeline) {
            Some(pipeline) => pipeline,
            None => return, // Pipeline not ready
        };

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, state.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(
            (width + workgroup_size.0 - 1) / workgroup_size.0,
            (height + workgroup_size.1 - 1) / workgroup_size.1,
            1,
        );
    }

    /// Creates the bind group layout for the image processing shader.
    fn create_bind_group_layout(
        &self,
        device: &RenderDevice,
        format: GpuImageFormat,
    ) -> BindGroupLayout {
        device.create_bind_group_layout(
            "image_processing_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // Input Image (read-only texture)
                    bevy::render::render_resource::binding_types::texture_2d(TextureSampleType::Float { filterable: false }),
                    // Output Image (write-only storage texture)
                    bevy::render::render_resource::binding_types::texture_storage_2d(format.into(), StorageTextureAccess::WriteOnly),
                ),
            ),
        )
    }

    /// Creates the bind group for a specific processing task.
    fn create_bind_group(
        &self,
        device: &RenderDevice,
        layout: &BindGroupLayout,
        input_image: &GpuImage,
        output_image: &GpuImage,
    ) -> BindGroup {
        device.create_bind_group(
            "image_processing_bind_group",
            layout,
            &BindGroupEntries::sequential((
                &input_image.texture_view,
                &output_image.texture_view,
            )),
        )
    }
} 