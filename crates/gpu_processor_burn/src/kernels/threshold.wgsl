@group(0) @binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0) @binding(2)
var<storage, read_write> info: array<u32>; // Contains width, height

@group(0) @binding(3)
var<storage, read_write> params: array<{{ elem }}>; // Contains threshold value, max_value

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

    let threshold_value = params[0];
    let max_value = params[1];

    let input_index = global_id.y * width + global_id.x;
    let pixel_value = input[input_index];

    // Apply threshold: pixel > threshold ? max_value : 0
    let output_value = select(0.0, max_value, pixel_value > threshold_value);

    output[input_index] = output_value;
} 