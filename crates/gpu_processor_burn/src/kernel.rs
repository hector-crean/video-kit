use burn_wgpu::{
    CubeTensor, FloatElement, KernelSource, SourceKernel,
    SourceTemplate, WgpuRuntime, kernel_source,
};
use cubecl::{CubeCount, CubeDim, KernelId, server::Bindings};
use derive_new::new;
use std::marker::PhantomData;

// Source all our kernels
kernel_source!(OutlineKernelRaw, "./kernels/outline.wgsl");
kernel_source!(GaussianBlurKernelRaw, "./kernels/gaussian_blur.wgsl");
kernel_source!(ThresholdKernelRaw, "./kernels/threshold.wgsl");
kernel_source!(MorphologyKernelRaw, "./kernels/morphology.wgsl");
kernel_source!(FusedPipelineKernelRaw, "./kernels/fused_pipeline.wgsl");

// Define kernel types with cube information
#[derive(new, Debug)]
struct OutlineKernel<F: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<F>,
}

#[derive(new, Debug)]
struct GaussianBlurKernel<F: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<F>,
}

#[derive(new, Debug)]
struct ThresholdKernel<F: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<F>,
}

#[derive(new, Debug)]
struct MorphologyKernel<F: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<F>,
}

#[derive(new, Debug)]
struct FusedPipelineKernel<F: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<F>,
}

// Implement KernelSource for all kernels
impl<F: FloatElement> KernelSource for OutlineKernel<F> {
    fn source(&self) -> SourceTemplate {
        OutlineKernelRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", F::type_name())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

impl<F: FloatElement> KernelSource for GaussianBlurKernel<F> {
    fn source(&self) -> SourceTemplate {
        GaussianBlurKernelRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", F::type_name())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

impl<F: FloatElement> KernelSource for ThresholdKernel<F> {
    fn source(&self) -> SourceTemplate {
        ThresholdKernelRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", F::type_name())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

impl<F: FloatElement> KernelSource for MorphologyKernel<F> {
    fn source(&self) -> SourceTemplate {
        MorphologyKernelRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", F::type_name())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

impl<F: FloatElement> KernelSource for FusedPipelineKernel<F> {
    fn source(&self) -> SourceTemplate {
        FusedPipelineKernelRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", F::type_name())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

// Kernel execution functions
/// Execute the outline extraction kernel on the given tensor.
pub fn extract_outline_kernel<F: FloatElement>(
    input: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
    let shape = input.shape.clone();
    let height = shape.dims[shape.num_dims() - 2];
    let width = shape.dims[shape.num_dims() - 1];
    
    let buffer = input
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let kernel = OutlineKernel::<F>::new(cube_dim);
    let info = [width as u32, height as u32];
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let cubes_needed_in_x = f32::ceil(width as f32 / cube_dim.x as f32) as u32;
    let cubes_needed_in_y = f32::ceil(height as f32 / cube_dim.y as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    input.client.execute(
        Box::new(SourceKernel::new(kernel, cube_dim)),
        cube_count,
        Bindings::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            info_handle.binding(),
        ]),
    );

    output
}

/// Execute Gaussian blur kernel
pub fn gaussian_blur_kernel<F: FloatElement>(
    input: CubeTensor<WgpuRuntime>,
    radius: u32,
) -> CubeTensor<WgpuRuntime> {
    let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
    let shape = input.shape.clone();
    let height = shape.dims[shape.num_dims() - 2];
    let width = shape.dims[shape.num_dims() - 1];
    
    let buffer = input
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let kernel = GaussianBlurKernel::<F>::new(cube_dim);
    let info = [width as u32, height as u32, radius];
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let cubes_needed_in_x = f32::ceil(width as f32 / cube_dim.x as f32) as u32;
    let cubes_needed_in_y = f32::ceil(height as f32 / cube_dim.y as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    input.client.execute(
        Box::new(SourceKernel::new(kernel, cube_dim)),
        cube_count,
        Bindings::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            info_handle.binding(),
        ]),
    );

    output
}

/// Execute threshold kernel
pub fn threshold_kernel<F: FloatElement>(
    input: CubeTensor<WgpuRuntime>,
    threshold_value: f32,
    max_value: f32,
) -> CubeTensor<WgpuRuntime> {
    let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
    let shape = input.shape.clone();
    let height = shape.dims[shape.num_dims() - 2];
    let width = shape.dims[shape.num_dims() - 1];
    
    let buffer = input
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let kernel = ThresholdKernel::<F>::new(cube_dim);
    
    // Create info and params buffers
    let info = [width as u32, height as u32];
    let info_handle = input.client.create(bytemuck::cast_slice(&info));
    
    let params = [threshold_value, max_value];
    let params_handle = input.client.create(bytemuck::cast_slice(&params));

    let cubes_needed_in_x = f32::ceil(width as f32 / cube_dim.x as f32) as u32;
    let cubes_needed_in_y = f32::ceil(height as f32 / cube_dim.y as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    input.client.execute(
        Box::new(SourceKernel::new(kernel, cube_dim)),
        cube_count,
        Bindings::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            info_handle.binding(),
            params_handle.binding(),
        ]),
    );

    output
}

/// Execute morphology kernel (erosion or dilation)
pub fn morphology_kernel<F: FloatElement>(
    input: CubeTensor<WgpuRuntime>,
    kernel_size: u32,
    is_erosion: bool,
) -> CubeTensor<WgpuRuntime> {
    let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
    let shape = input.shape.clone();
    let height = shape.dims[shape.num_dims() - 2];
    let width = shape.dims[shape.num_dims() - 1];
    
    let buffer = input
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let kernel = MorphologyKernel::<F>::new(cube_dim);
    let operation_type = if is_erosion { 0u32 } else { 1u32 };
    let info = [width as u32, height as u32, kernel_size, operation_type];
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let cubes_needed_in_x = f32::ceil(width as f32 / cube_dim.x as f32) as u32;
    let cubes_needed_in_y = f32::ceil(height as f32 / cube_dim.y as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    input.client.execute(
        Box::new(SourceKernel::new(kernel, cube_dim)),
        cube_count,
        Bindings::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            info_handle.binding(),
        ]),
    );

    output
}

/// Execute fused pipeline kernel (blur + edge detection + threshold)
pub fn fused_pipeline_kernel<F: FloatElement>(
    input: CubeTensor<WgpuRuntime>,
    blur_radius: u32,
    threshold_value: f32,
    edge_threshold: f32,
) -> CubeTensor<WgpuRuntime> {
    let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
    let shape = input.shape.clone();
    let height = shape.dims[shape.num_dims() - 2];
    let width = shape.dims[shape.num_dims() - 1];
    
    let buffer = input
        .client
        .empty(shape.num_elements() * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        F::dtype(),
    );

    let kernel = FusedPipelineKernel::<F>::new(cube_dim);
    
    let info = [width as u32, height as u32, blur_radius];
    let info_handle = input.client.create(bytemuck::cast_slice(&info));
    
    let params = [threshold_value, edge_threshold];
    let params_handle = input.client.create(bytemuck::cast_slice(&params));

    let cubes_needed_in_x = f32::ceil(width as f32 / cube_dim.x as f32) as u32;
    let cubes_needed_in_y = f32::ceil(height as f32 / cube_dim.y as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, 1);

    input.client.execute(
        Box::new(SourceKernel::new(kernel, cube_dim)),
        cube_count,
        Bindings::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            info_handle.binding(),
            params_handle.binding(),
        ]),
    );

    output
} 