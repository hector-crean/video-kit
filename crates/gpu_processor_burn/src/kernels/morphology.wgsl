@group(0) @binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0) @binding(2)
var<storage, read_write> info: array<u32>; // Contains width, height, kernel_size, operation_type

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
    let kernel_size = i32(info[2]);
    let operation_type = info[3]; // 0 = erosion, 1 = dilation

    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(width) || y >= i32(height)) {
        return;
    }

    let radius = kernel_size / 2;
    var result_value = select(1.0, 0.0, operation_type == 0u); // Start with max for erosion, min for dilation

    // Apply morphological operation
    for (var ky = -radius; ky <= radius; ky++) {
        for (var kx = -radius; kx <= radius; kx++) {
            let pixel_val = get_pixel(x + kx, y + ky, width, height);
            
            if (operation_type == 0u) {
                // Erosion: find minimum
                result_value = min(result_value, pixel_val);
            } else {
                // Dilation: find maximum
                result_value = max(result_value, pixel_val);
            }
        }
    }

    let output_index = global_id.y * width + global_id.x;
    output[output_index] = result_value;
} 