@group(0) @binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0) @binding(2)
var<storage, read_write> info: array<u32>; // Contains width, height, blur_radius

@group(0) @binding(3)
var<storage, read_write> params: array<{{ elem }}>; // Contains threshold_value, edge_threshold

/// Safely get a pixel from the input buffer, returning 0.0 for out-of-bounds reads.
fn get_pixel(x: i32, y: i32, width: u32, height: u32) -> {{ elem }} {
    if (x < 0 || x >= i32(width) || y < 0 || y >= i32(height)) {
        return 0.0;
    }
    return input[u32(y) * width + u32(x)];
}

/// Calculate Gaussian weight
fn gaussian_weight(distance: f32, sigma: f32) -> f32 {
    let sigma_sq_2 = 2.0 * sigma * sigma;
    return exp(-(distance * distance) / sigma_sq_2);
}

/// Apply Gaussian blur to a pixel
fn apply_blur(x: i32, y: i32, width: u32, height: u32, radius: i32, sigma: f32) -> {{ elem }} {
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var ky = -radius; ky <= radius; ky++) {
        for (var kx = -radius; kx <= radius; kx++) {
            let pixel_val = get_pixel(x + kx, y + ky, width, height);
            let distance = sqrt(f32(kx * kx + ky * ky));
            let weight = gaussian_weight(distance, sigma);
            
            sum += pixel_val * weight;
            weight_sum += weight;
        }
    }

    return sum / weight_sum;
}

/// Apply Sobel edge detection
fn apply_sobel(x: i32, y: i32, width: u32, height: u32) -> {{ elem }} {
    // Sobel Gx kernel
    let gx = -1.0 * get_pixel(x - 1, y - 1, width, height) + 1.0 * get_pixel(x + 1, y - 1, width, height)
           -2.0 * get_pixel(x - 1, y,     width, height) + 2.0 * get_pixel(x + 1, y,     width, height)
           -1.0 * get_pixel(x - 1, y + 1, width, height) + 1.0 * get_pixel(x + 1, y + 1, width, height);

    // Sobel Gy kernel
    let gy = -1.0 * get_pixel(x - 1, y - 1, width, height) - 2.0 * get_pixel(x, y - 1, width, height) - 1.0 * get_pixel(x + 1, y - 1, width, height)
           +1.0 * get_pixel(x - 1, y + 1, width, height) + 2.0 * get_pixel(x, y + 1, width, height) + 1.0 * get_pixel(x + 1, y + 1, width, height);

    // Calculate the gradient magnitude
    return sqrt(gx * gx + gy * gy);
}

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = info[0];
    let height = info[1];
    let blur_radius = i32(info[2]);

    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(width) || y >= i32(height)) {
        return;
    }

    let threshold_value = params[0];
    let edge_threshold = params[1];

    // Step 1: Apply Gaussian blur (noise reduction)
    let sigma = f32(blur_radius) / 3.0;
    let blurred = apply_blur(x, y, width, height, blur_radius, sigma);
    
    // Temporarily store blurred value for edge detection
    let current_index = global_id.y * width + global_id.x;
    input[current_index] = blurred;
    
    // Synchronize to ensure all threads have updated their blur values
    workgroupBarrier();
    
    // Step 2: Apply edge detection on blurred image
    let edge_magnitude = apply_sobel(x, y, width, height);
    
    // Step 3: Apply threshold to create binary edge map
    let final_result = select(0.0, 1.0, edge_magnitude > edge_threshold);

    output[current_index] = final_result;
} 