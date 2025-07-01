@group(0) @binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0) @binding(2)
var<storage, read_write> info: array<u32>; // Contains width, height

/// Safely get a pixel from the input buffer, returning 0.0 for out-of-bounds reads.
fn get_pixel(x: i32, y: i32, width: u32, height: u32) -> {{ elem }} {
    if (x < 0 || x >= i32(width) || y < 0 || y >= i32(height)) {
        return 0.0;
    }
    return input[u32(y) * width + u32(x)];
}

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = info[0];
    let height = info[1];

    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(width) || y >= i32(height)) {
        return;
    }

    // Sobel Gx kernel
    let gx = -1.0 * get_pixel(x - 1, y - 1, width, height) + 1.0 * get_pixel(x + 1, y - 1, width, height)
           -2.0 * get_pixel(x - 1, y,     width, height) + 2.0 * get_pixel(x + 1, y,     width, height)
           -1.0 * get_pixel(x - 1, y + 1, width, height) + 1.0 * get_pixel(x + 1, y + 1, width, height);

    // Sobel Gy kernel
    let gy = -1.0 * get_pixel(x - 1, y - 1, width, height) - 2.0 * get_pixel(x, y - 1, width, height) - 1.0 * get_pixel(x + 1, y - 1, width, height)
           +1.0 * get_pixel(x - 1, y + 1, width, height) + 2.0 * get_pixel(x, y + 1, width, height) + 1.0 * get_pixel(x + 1, y + 1, width, height);

    // Calculate the gradient magnitude
    let magnitude = sqrt(gx * gx + gy * gy);
    
    let output_index = global_id.y * width + global_id.x;
    output[output_index] = magnitude;
} 