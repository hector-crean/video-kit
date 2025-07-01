@group(0) @binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0) @binding(2)
var<storage, read_write> info: array<u32>; // Contains width, height, radius

/// Safely get a pixel from the input buffer, returning 0.0 for out-of-bounds reads.
fn get_pixel(x: i32, y: i32, width: u32, height: u32) -> {{ elem }} {
    if (x < 0 || x >= i32(width) || y < 0 || y >= i32(height)) {
        return 0.0;
    }
    return input[u32(y) * width + u32(x)];
}

/// Calculate Gaussian weight
fn gaussian_weight(x: f32, sigma: f32) -> f32 {
    let sigma_sq_2 = 2.0 * sigma * sigma;
    return exp(-(x * x) / sigma_sq_2);
}

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = info[0];
    let height = info[1];
    let radius = i32(info[2]);

    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(width) || y >= i32(height)) {
        return;
    }

    let sigma = f32(radius) / 3.0; // Standard sigma relationship
    var sum = 0.0;
    var weight_sum = 0.0;

    // Apply Gaussian blur in both dimensions
    for (var ky = -radius; ky <= radius; ky++) {
        for (var kx = -radius; kx <= radius; kx++) {
            let pixel_val = get_pixel(x + kx, y + ky, width, height);
            let distance = sqrt(f32(kx * kx + ky * ky));
            let weight = gaussian_weight(distance, sigma);
            
            sum += pixel_val * weight;
            weight_sum += weight;
        }
    }

    let output_index = global_id.y * width + global_id.x;
    output[output_index] = sum / weight_sum;
} 