// script.js

// --- Global WebGPU Variables ---
let device;
let context;
let presentationFormat;
let pipeline;
let vertexBuffer; // Stores vertices for one shape (e.g., an asteroid segment)
let uniformBuffer; // For camera/projection matrix
let objectModelMatricesBuffer; // For individual object model matrices (no color needed per object)
let bindGroup;

const OBJECT_COUNT = 150; // More "stars" / "asteroids"
const OBJECT_SCALE_MIN = 0.05;
const OBJECT_SCALE_MAX = 0.2;
const LINE_COLOR = [0.0, 1.0, 0.0, 1.0]; // Bright green for classic vector look

// Simple object representation
let objects = []; // Each object will have position, rotation, scale, speed

// Camera properties
const camera = {
    position: [0.0, 0.0, 0.0],
    speed: 0.1, // Faster movement
    projectionMatrix: new Float32Array(16), // 4x4 matrix
    viewMatrix: new Float32Array(16) // 4x4 matrix
};

// --- Matrix Math Utilities (Simplified - Same as previous example) ---
function mat4_identity(out) {
    out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
}

function mat4_perspective(out, fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = (far + near) / (near - far); out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = (2 * far * near) / (near - far); out[15] = 0;
    return out;
}

function mat4_lookAt(out, eye, center, up) {
    let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
    let eyex = eye[0];
    let eyey = eye[1];
    let eyez = eye[2];
    let upx = up[0];
    let upy = up[1];
    let upz = up[2];
    let centerx = center[0];
    let centery = center[1];
    let centerz = center[2];

    if (
        Math.abs(eyex - centerx) < 0.000001 &&
        Math.abs(eyey - centery) < 0.000001 &&
        Math.abs(eyez - centerz) < 0.000001
    ) {
        return mat4_identity(out);
    }

    z0 = eyex - centerx;
    z1 = eyey - centery;
    z2 = eyez - centerz;

    len = 1 / Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
    z0 *= len;
    z1 *= len;
    z2 *= len;

    x0 = upy * z2 - upz * z1;
    x1 = upz * z0 - upx * z2;
    x2 = upx * z1 - upy * z0;
    len = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    if (len === 0) {
        x0 = 0;
        x1 = 0;
        x2 = 0;
    } else {
        len = 1 / len;
        x0 *= len;
        x1 *= len;
        x2 *= len;
    }

    y0 = z1 * x2 - z2 * x1;
    y1 = z2 * x0 - z0 * x2;
    y2 = z0 * x1 - z1 * x0;

    len = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
    if (len === 0) {
        y0 = 0;
        y1 = 0;
        y2 = 0;
    } else {
        len = 1 / len;
        y0 *= len;
        y1 *= len;
        y2 *= len;
    }

    out[0] = x0;
    out[1] = y0;
    out[2] = z0;
    out[3] = 0;
    out[4] = x1;
    out[5] = y1;
    out[6] = z1;
    out[7] = 0;
    out[8] = x2;
    out[9] = y2;
    out[10] = z2;
    out[11] = 0;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1;

    return out;
}

function mat4_multiply(out, a, b) {
    let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    let a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    let a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    let a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
}

function mat4_translation(out, x, y, z) {
    out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
    out[12] = x; out[13] = y; out[14] = z; out[15] = 1;
    return out;
}

function mat4_scale(out, x, y, z) {
    out[0] = x; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = y; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = z; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
}

function mat4_rotateZ(out, rad) {
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    out[0] = c; out[1] = s; out[2] = 0; out[3] = 0;
    out[4] = -s; out[5] = c; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
}

// Helper to create model matrix (translation * rotation * scale)
const tempMat1 = new Float32Array(16);
const tempMat2 = new Float32Array(16);
function getModelMatrix(position, rotationZ, scale) {
    mat4_translation(tempMat1, position[0], position[1], position[2]);
    mat4_rotateZ(tempMat2, rotationZ);
    mat4_multiply(tempMat1, tempMat1, tempMat2); // Translate then Rotate
    mat4_scale(tempMat2, scale, scale, scale);
    const modelMatrix = new Float32Array(16);
    mat4_multiply(modelMatrix, tempMat1, tempMat2); // Apply scaling last
    return modelMatrix;
}

// --- WebGPU Initialization ---
async function initWebGPU() {
    if (!navigator.gpu) {
        alert("WebGPU not supported on your browser/device. Please try Chrome 113+ on Windows/macOS/ChromeOS, or a nightly build of Firefox/Safari.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert("No WebGPU adapter found.");
        return;
    }

    device = await adapter.requestDevice();
    device.lost.then(() => {
        console.error('WebGPU device was lost!');
        alert('WebGPU device was lost! Please reload the page.');
    });

    const canvas = document.getElementById('webgpu-canvas');
    context = canvas.getContext('webgpu');

    presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: device,
        format: presentationFormat,
        alphaMode: 'premultiplied', // Lines can have transparency if needed, or opaque
    });

    createGeometry();
    createUniformBuffers();
    await createPipeline(); // Await pipeline creation as it's async
    createBindGroups();

    generateObjects();

    // Start render loop
    requestAnimationFrame(renderLoop);
    setupInputHandling();
    window.addEventListener('resize', onWindowResize);
    onWindowResize(); // Initial call to set aspect ratio
}

function createGeometry() {
    // Define vertices for a simple star-like shape (or simple asteroid outline)
    // This is a line list: (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)...
    // Each pair of points defines a line segment.
    const vertices = new Float32Array([
        // Simple Asteroid-like shape (6 points, 6 lines to close it)
        // Vertices ordered for line-list: (start_x, start_y, start_z, end_x, end_y, end_z)
        // Each pair of (x,y,z) is a vertex. Each consecutive 2 vertices form a line.
        -0.5, -0.2, 0.0, // V0
         0.0, -0.5, 0.0, // V1
         0.0, -0.5, 0.0, // V1 (duplicate to start new segment from same point)
         0.5, -0.2, 0.0, // V2
         0.5, -0.2, 0.0, // V2
         0.3,  0.3, 0.0, // V3
         0.3,  0.3, 0.0, // V3
        -0.2,  0.5, 0.0, // V4
        -0.2,  0.5, 0.0, // V4
        -0.5,  0.2, 0.0, // V5
        -0.5,  0.2, 0.0, // V5
        -0.5, -0.2, 0.0, // V0 (closing segment)
    ]);

    vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();
}

function createUniformBuffers() {
    // Camera/Projection Uniform Buffer (single matrix for view-projection)
    uniformBuffer = device.createBuffer({
        size: 4 * 16, // 16 floats * 4 bytes/float
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Object Model Matrices Uniform Buffer
    // Each object needs a 4x4 model matrix (16 floats)
    objectModelMatricesBuffer = device.createBuffer({
        size: OBJECT_COUNT * (16 * 4), // 16 floats * 4 bytes/float per matrix * OBJECT_COUNT
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

async function createPipeline() {
    const shaderModule = device.createShaderModule({
        code: `
            // Camera (view-projection) matrix
            struct CameraUniform {
                viewProjectionMatrix : mat4x4<f32>,
            };
            @group(0) @binding(0) var<uniform> camera : CameraUniform;

            // Model matrices for each object
            struct ModelMatrices {
                matrices : array<mat4x4<f32>, ${OBJECT_COUNT}>,
            };
            @group(0) @binding(1) var<uniform> models : ModelMatrices;

            // Global line color for vector look (Now only used in fragment shader, but declared in group 0)
            struct LineColorUniform {
                color : vec4<f32>,
            };
            // Note: This uniform is used to pass the color from vertex to fragment shader via 'out.vertex_color'.
            // Therefore, it needs visibility in the VERTEX stage.
            @group(0) @binding(2) var<uniform> lineColorUniform : LineColorUniform; // Renamed to avoid confusion

            struct VertexInput {
                @location(0) position : vec3<f32>,
            };

            struct VertexOutput {
                @builtin(position) clip_position : vec4<f32>,
                @location(0) vertex_color : vec4<f32>, // Pass color to fragment shader
            };

            @vertex
            fn vs_main(
                in: VertexInput,
                @builtin(instance_index) instance_idx : u32
            ) -> VertexOutput {
                var out: VertexOutput;
                let modelMatrix = models.matrices[instance_idx];
                out.clip_position = camera.viewProjectionMatrix * modelMatrix * vec4<f32>(in.position, 1.0);
                out.vertex_color = lineColorUniform.color; // All lines are the same color, sourced from the uniform
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return in.vertex_color;
            }
        `,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0, // For camera uniform
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' },
            },
            {
                binding: 1, // For object model matrices uniform array
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: OBJECT_COUNT * 16 * 4 },
            },
            {
                binding: 2, // For global line color - FIX APPLIED HERE
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, // Now visible to both
                buffer: { type: 'uniform' },
            },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    pipeline = await device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [{
                arrayStride: 3 * 4, // 3 floats * 4 bytes/float = 12 bytes
                attributes: [{
                    shaderLocation: 0, // Corresponds to @location(0) in VS
                    offset: 0,
                    format: 'float32x3', // Position (x, y, z)
                }],
            }],
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{
                format: presentationFormat,
                // Add blending for glowing effect (additive)
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one',
                        operation: 'add',
                    },
                },
            }],
        },
        primitive: {
            topology: 'line-list', // Now we're drawing lines!
            // rasterizationState: {
            //     cullMode: 'back', // Not strictly necessary for lines
            // }
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    });

    // Create Depth Texture for depth buffer
    const canvas = document.getElementById('webgpu-canvas');
    depthTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthTextureView = depthTexture.createView();
}

let depthTexture;
let depthTextureView;
let lineColorBuffer; // New uniform buffer for line color

function onWindowResize() {
    if (!device) return;

    const canvas = document.getElementById('webgpu-canvas');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;

    context.configure({
        device: device,
        format: presentationFormat,
        alphaMode: 'premultiplied',
        size: [canvas.width, canvas.height],
    });

    if (depthTexture) {
        depthTexture.destroy();
    }
    depthTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthTextureView = depthTexture.createView();

    updateProjectionMatrix(canvas.width / canvas.height);
}


function createBindGroups() {
    // Create the line color uniform buffer once
    lineColorBuffer = device.createBuffer({
        size: 4 * 4, // 4 floats for vec4, 4 bytes/float
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(lineColorBuffer.getMappedRange()).set(LINE_COLOR);
    lineColorBuffer.unmap();

    bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: objectModelMatricesBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: lineColorBuffer,
                },
            },
        ],
    });
}

function generateObjects() {
    for (let i = 0; i < OBJECT_COUNT; i++) {
        objects.push({
            position: [
                (Math.random() - 0.5) * 40, // x
                (Math.random() - 0.5) * 40, // y
                (Math.random() - 0.5) * 200 - 100 // z (spread out more along z-axis)
            ],
            rotationZ: Math.random() * Math.PI * 2, // Initial random rotation
            rotationSpeed: (Math.random() - 0.5) * 0.05, // Spin speed
            scale: OBJECT_SCALE_MIN + Math.random() * (OBJECT_SCALE_MAX - OBJECT_SCALE_MIN),
            speed: 0.1 + Math.random() * 0.1 // Varying speeds
        });
    }
}

// --- Update Logic ---
let keysPressed = {};
function setupInputHandling() {
    document.addEventListener('keydown', (e) => {
        keysPressed[e.key.toLowerCase()] = true;
    });
    document.addEventListener('keyup', (e) => {
        keysPressed[e.key.toLowerCase()] = false;
    });
}

function updateCamera() {
    const moveSpeed = camera.speed;
    // For simplicity, we're not rotating the camera's forward/right vectors based on its orientation.
    // Movement is relative to world axes.
    let moved = false;

    if (keysPressed['w'] || keysPressed['arrowup']) {
        camera.position[2] -= moveSpeed; // Move forward along Z
        moved = true;
    }
    if (keysPressed['s'] || keysPressed['arrowdown']) {
        camera.position[2] += moveSpeed; // Move backward along Z
        moved = true;
    }
    if (keysPressed['a'] || keysPressed['arrowleft']) {
        camera.position[0] -= moveSpeed; // Strafe left along X
        moved = true;
    }
    if (keysPressed['d'] || keysPressed['arrowright']) {
        camera.position[0] += moveSpeed; // Strafe right along X
        moved = true;
    }

    if (moved) {
        updateViewMatrix();
    }
}

function updateObjects(deltaTime) {
    for (let i = 0; i < objects.length; i++) {
        const obj = objects[i];
        // Move objects towards the camera (simulating camera moving forward)
        obj.position[2] += obj.speed * deltaTime * 0.01; // Adjust speed by delta time
        obj.rotationZ += obj.rotationSpeed * deltaTime * 0.01; // Spin objects

        // If object goes too far behind the camera, reset it to the front
        // The camera is at camera.position[2], looking towards negative Z.
        // Objects "pass" the camera when their Z position is greater than camera.position[2].
        // We reset them far in front of the camera.
        const resetThreshold = camera.position[2] + 5; // A bit in front of the camera
        const spawnDistance = 200; // How far in front to re-spawn
        if (obj.position[2] > resetThreshold) {
            obj.position[2] = camera.position[2] - spawnDistance + (Math.random() * 10 - 5); // Add slight random variation
            obj.position[0] = (Math.random() - 0.5) * 40;
            obj.position[1] = (Math.random() - 0.5) * 40;
            obj.rotationZ = Math.random() * Math.PI * 2;
            obj.rotationSpeed = (Math.random() - 0.5) * 0.05;
            obj.scale = OBJECT_SCALE_MIN + Math.random() * (OBJECT_SCALE_MAX - OBJECT_SCALE_MIN);
            obj.speed = 0.1 + Math.random() * 0.1;
        }
    }
}

function updateProjectionMatrix(aspect) {
    const fov = (60 * Math.PI) / 180; // Wider FOV for "speed"
    const near = 0.1;
    const far = 200.0; // Extend far plane
    mat4_perspective(camera.projectionMatrix, fov, aspect, near, far);
}

function updateViewMatrix() {
    const eye = camera.position;
    // The "center" point defines where the camera is looking.
    // For a simple first-person "flying" game, it's just a point directly in front of the camera.
    const center = [camera.position[0], camera.position[1], camera.position[2] - 1]; // Look slightly forward
    const up = [0, 1, 0]; // Y-axis is up
    mat4_lookAt(camera.viewMatrix, eye, center, up);
}

// Combine view and projection for the uniform buffer
const viewProjectionMatrix = new Float32Array(16);
function updateCameraUniformBuffer() {
    mat4_multiply(viewProjectionMatrix, camera.projectionMatrix, camera.viewMatrix);
    device.queue.writeBuffer(uniformBuffer, 0, viewProjectionMatrix);
}

// Prepare object model matrices for the uniform buffer
const objectModelMatricesArray = new Float32Array(OBJECT_COUNT * 16); // 16 floats per matrix
function updateObjectModelMatricesBuffer() {
    let offset = 0;
    for (let i = 0; i < objects.length; i++) {
        const obj = objects[i];
        const modelMatrix = getModelMatrix(obj.position, obj.rotationZ, obj.scale);
        objectModelMatricesArray.set(modelMatrix, offset);
        offset += 16;
    }
    device.queue.writeBuffer(objectModelMatricesBuffer, 0, objectModelMatricesArray);
}

// --- Render Loop ---
let lastTime = 0;
function renderLoop(currentTime) {
    if (!device) return;

    const deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    updateCamera();
    updateObjects(deltaTime);
    updateCameraUniformBuffer();
    updateObjectModelMatricesBuffer();

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor = {
        colorAttachments: [{
            view: textureView,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, // Pure black background
            loadOp: 'clear',
            storeOp: 'store',
        }],
        depthStencilAttachment: {
            view: depthTextureView,
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    // The number of vertices to draw. Our asteroid shape has 12 vertices (6 lines * 2 vertices/line).
    passEncoder.draw(12, OBJECT_COUNT); // Fixed to 12 vertices for the asteroid shape
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(renderLoop);
}

// --- Start the Application ---
window.onload = initWebGPU;