//
// Grayscale Compute Shader
// Converts an input image to grayscale.
//

// The input image, read-only.
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

// The output image, write-only.
@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

// Main compute shader function.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get the dimensions of the output texture.
    let output_size = textureDimensions(output_texture);

    // Prevent out-of-bounds access.
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }

    // Load the color from the input texture.
    let input_color = textureLoad(input_texture, global_id.xy, 0);
    let input_rgb = input_color.rgb;

    // Apply the luminance formula to calculate the grayscale value.
    // NTSC formula: 0.299*R + 0.587*G + 0.114*B
    let gray = dot(input_rgb, vec3<f32>(0.299, 0.587, 0.114));

    // Create the final grayscale color.
    let output_color = vec4<f32>(gray, gray, gray, input_color.a);

    // Write the result to the output texture.
    textureStore(output_texture, global_id.xy, output_color);
} 