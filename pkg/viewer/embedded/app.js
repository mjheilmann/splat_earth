// 3D Gaussian Splat Spheroid Terrain Renderer with Spatial Chunking & Hierarchical LOD
(function() {
    'use strict';

    // State
    const state = {
        splats: null,
        splatCount: 0,
        activeSplatCount: 0,
        spheroid: { radius: 100.0, flattening: 0.05 },
        scaleMultiplier: 1.0,
        thicknessMultiplier: 0.35,
        alphaBoost: 1.0,
        colorMode: 0,
        kernelType: 0,          // 0 = Flat-Top Plateau (Solid & Crisp), 1 = Super-Gaussian, 2 = Standard Gaussian
        sunAzimuth: 45.0,       // Sun angle around horizon (0-360 deg)
        sunElevation: 35.0,     // Sun altitude above horizon (-10 to 90 deg)
        sunIntensity: 1.25,     // Direct sunlight intensity
        ambientIntensity: 0.38, // Hemispheric skylight intensity
        shadingEnabled: true,   // Dynamic PBR sun lighting & normal shading
        hazeDensity: 0.50,      // Atmospheric Rayleigh haze density
        showGrid: true,
        depthSort: false,
        lodBias: 1.0,
        cameraMode: 'fly', // 'fly' or 'orbit'
        flightSpeed: 50.0,
        altitude: 100.0,
        chunks: [],
        chunkManifests: [],
        activeChunksCount: 0,
        camera: {
            pos: [0, -240, 120],
            target: [0, 0, 0],
            quat: [1, 0, 0, 0], // Camera orientation quaternion [w, x, y, z] for 6-DOF flight
            yaw: Math.PI * 0.5,
            pitch: -0.45,
            orbitDistance: 250,
            orbitTheta: 0,
            orbitPhi: 0.45,
            fov: 60 * Math.PI / 180,
            near: 0.001,
            far: 1000000.0,
        },
        keys: {},
        isDragging: false,
        lastMouseX: 0,
        lastMouseY: 0,
        fps: 60,
        lastTime: performance.now(),
        frameCount: 0,
        fpsTimer: performance.now(),
        lastLODUpdate: 0
    };

    // DOM Elements
    const canvas = document.getElementById('gl-canvas');
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' });

    if (!gl) {
        alert('WebGL 2.0 is required for 3D Gaussian Splat rendering.');
        return;
    }

    // Shaders for 3D Gaussian Splatting with GPU Horizon & Frustum Culling
    const splatVS = `#version 300 es
    precision highp float;

    layout(location = 0) in vec2 a_quad;        // (-2,-2) to (2,2)
    layout(location = 1) in vec3 a_center;      // Splat center X,Y,Z
    layout(location = 2) in vec3 a_scale;       // Splat scale Sx,Sy,Sz
    layout(location = 3) in vec4 a_rotation;    // Quaternion W,X,Y,Z
    layout(location = 4) in vec4 a_color;       // RGBA uint8 (normalized to 0..1)

    uniform mat4 u_view;
    uniform mat4 u_projection;
    uniform vec3 u_camPos;
    uniform vec2 u_focal;         // fx, fy
    uniform vec2 u_viewport;      // width, height
    uniform float u_scaleMult;
    uniform float u_thicknessMult;
    uniform int u_colorMode;

    out vec2 v_uv;
    out vec4 v_color;
    out vec3 v_normal;
    out vec3 v_worldPos;
    out float v_altitude;

    // Helper: Quaternion (W,X,Y,Z) to Column-Major Rotation Matrix
    mat3 quatToMat(vec4 q) {
        float w = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;

        // GLSL constructor mat3(col0, col1, col2) is COLUMN-MAJOR
        vec3 col0 = vec3(
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y + w*z),
            2.0*(x*z - w*y)
        );
        vec3 col1 = vec3(
            2.0*(x*y - w*z),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z + w*x)
        );
        vec3 col2 = vec3(
            2.0*(x*z + w*y),
            2.0*(y*z - w*x),
            1.0 - 2.0*(x*x + y*y)
        );

        return mat3(col0, col1, col2);
    }

    void main() {
        vec3 center = a_center;

        // Rotation matrix from normal-aligned quaternion
        mat3 R = quatToMat(a_rotation);
        // --- Fast GPU Planetary Horizon Culling ---
        // For a spheroid centered at the origin (0,0,0), horizon angle theta_h = acos(clamp(r_planet / distCam, 0.0, 1.0)).
        // A splat is visible if angle(u_cam, u_point) <= theta_h + margin.
        float camDist = length(u_camPos);
        float ptDist = length(center);
        if (camDist > ptDist && ptDist > 1.0) {
            float cosHoriz = ptDist / camDist;
            float horizonAngle = acos(clamp(cosHoriz, 0.0, 1.0));
            float minCos = cos(min(3.14159, horizonAngle + 0.08));
            if (dot(u_camPos / camDist, center / ptDist) < minCos) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }
        }

        // Transform center to camera view space
        vec4 pView = u_view * vec4(center, 1.0);
        if (pView.z >= -0.00001) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }

        vec3 scale = vec3(a_scale.xy * u_scaleMult, a_scale.z * u_thicknessMult);
        mat3 S = mat3(
            scale.x, 0.0, 0.0,
            0.0, scale.y, 0.0,
            0.0, 0.0, scale.z
        );
        mat3 M = R * S;
        mat3 Sigma = M * transpose(M); // 3D Covariance Ellipsoid

        mat3 W = mat3(u_view);

        // Jacobian of perspective projection at pView
        float z = pView.z;
        float z2 = z * z;
        mat3x2 J = mat3x2(
            u_focal.x / z, 0.0,
            0.0, u_focal.y / z,
            -(u_focal.x * pView.x) / z2, -(u_focal.y * pView.y) / z2
        );

        // 2D Screen Space Covariance: V = J * W * Sigma * W^T * J^T
        mat3 W_Sigma_WT = W * Sigma * transpose(W);
        mat2 V = J * W_Sigma_WT * transpose(J);

        // Anti-aliasing low-pass filter
        V[0][0] += 0.3;
        V[1][1] += 0.3;

        // Eigen decomposition of 2x2 covariance matrix V:
        float a = V[0][0];
        float b = V[0][1];
        float c = V[1][1];

        float trace = a + c;
        float det = a * c - b * b;
        float term = sqrt(max(0.0001, (a - c) * (a - c) + 4.0 * b * b));

        float lambda1 = max(0.1, (trace + term) * 0.5);
        float lambda2 = max(0.1, (trace - term) * 0.5);

        // Eigenvector for lambda1
        vec2 ev1 = vec2(b, lambda1 - a);
        if (length(ev1) < 1e-4) {
            ev1 = vec2(1.0, 0.0);
        } else {
            ev1 = normalize(ev1);
        }
        vec2 ev2 = vec2(-ev1.y, ev1.x);

        // Splat radii in pixels (3 standard deviations)
        float r1 = min(4096.0, 3.0 * sqrt(lambda1));
        float r2 = min(4096.0, 3.0 * sqrt(lambda2));

        // Offset in screen pixels
        vec2 pixelOffset = a_quad.x * ev1 * (r1 / 3.0) + a_quad.y * ev2 * (r2 / 3.0);
        vec2 ndcOffset = (pixelOffset / u_viewport) * 2.0;

        vec4 clipPos = u_projection * pView;
        clipPos.xy += ndcOffset * clipPos.w;

        // Fast GPU Frustum Culling
        if (abs(clipPos.x) > clipPos.w * 1.5 || abs(clipPos.y) > clipPos.w * 1.5 || clipPos.z < -clipPos.w || clipPos.z > clipPos.w * 1.1) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }

        gl_Position = clipPos;
        v_uv = a_quad; // (-2,-2) to (2,2)
        v_color = a_color;
        // True physical terrain surface normal from slope quaternion
        v_normal = normalize(R[2]);
        v_worldPos = center;
        v_altitude = length(center);
    }
    `;

    const splatFS = `#version 300 es
    precision highp float;

    in vec2 v_uv;
    in vec4 v_color;
    in vec3 v_normal;
    in vec3 v_worldPos;
    in float v_altitude;

    uniform vec3 u_camPos;
    uniform vec3 u_sunDir;
    uniform float u_sunIntensity;
    uniform float u_ambientIntensity;
    uniform float u_hazeDensity;
    uniform bool u_shadingEnabled;
    uniform float u_alphaBoost;
    uniform int u_colorMode;
    uniform int u_kernelType;
    uniform float u_spheroidRadius;

    out vec4 fragColor;

    vec3 colormapTurbo(float x) {
        x = clamp(x, 0.0, 1.0);
        vec4 r = vec4(0.1357, 4.6153, -42.6603, 132.131);
        vec4 g = vec4(0.0914, 2.1941, 4.8429, -14.185);
        vec4 b = vec4(0.1066, 12.587, -67.4347, 123.654);
        return clamp(vec3(
            r.x + x * (r.y + x * (r.z + x * r.w)),
            g.x + x * (g.y + x * (g.z + x * g.w)),
            b.x + x * (b.y + x * (b.z + x * b.w))
        ), 0.0, 1.0);
    }

    void main() {
        // Radially symmetric 2D Gaussian falloff (invariant to screen-space billboard rotation)
        float d = -0.5 * dot(v_uv, v_uv);
        if (d < -4.5) {
            discard; // Beyond 3 sigma
        }

        float alpha = exp(d) * v_color.a * u_alphaBoost;
        if (alpha < 0.02) {
            discard;
        }

        vec3 baseColor = v_color.rgb;

        if (u_colorMode == 1) {
            // Elevation Colormap
            float relH = (v_altitude - u_spheroidRadius) / max(1.0, u_spheroidRadius * 0.1);
            baseColor = colormapTurbo(relH * 0.8 + 0.2);
        } else if (u_colorMode == 2) {
            // Surface Normals Debug
            fragColor = vec4((v_normal * 0.5 + 0.5) * alpha, alpha);
            return;
        } else if (u_colorMode == 3) {
            // Gaussian Footprint Debug
            float ring = fract(length(v_uv) * 1.5);
            if (ring < 0.15 || length(v_uv) > 1.8) {
                baseColor = vec3(0.0, 1.0, 0.8);
            } else {
                baseColor = v_color.rgb * 0.5;
            }
        }

        // --- High-Frequency Micro-Relief Noise inside Splats ---
        // Adds subtle procedural rock/strata micro-grain across each wafer
        float grain1 = sin(v_uv.x * 9.42 + v_uv.y * 6.28) * cos(v_uv.x * 6.28 - v_uv.y * 12.56);
        float grain2 = sin(v_uv.x * 18.84 - v_uv.y * 15.70) * 0.5;
        float microRelief = (grain1 + grain2) * 0.065;
        baseColor = clamp(baseColor * (1.0 + microRelief), 0.0, 1.0);

        if (!u_shadingEnabled && u_colorMode == 0) {
            fragColor = vec4(baseColor * alpha, alpha);
            return;
        }

        // --- Dynamic Sun Lighting & Normal Shading ---
        vec3 N = normalize(v_normal);
        vec3 L = normalize(u_sunDir);
        vec3 V = normalize(u_camPos - v_worldPos);

        // Half-Lambert wrap diffuse for realistic soft mountain penumbra
        float NdotL = dot(N, L);
        float wrapDiffuse = clamp((NdotL + 0.20) / 1.20, 0.0, 1.0);

        // Hemispherical Ambient Skylight (cool Rayleigh blue from sky, warm ground bounce)
        vec3 upNorm = normalize(v_worldPos);
        float hemi = dot(N, upNorm) * 0.5 + 0.5;
        vec3 skyAmbient = mix(vec3(0.10, 0.14, 0.22), vec3(0.38, 0.46, 0.58), hemi) * u_ambientIntensity;

        // Direct warm golden sunlight
        vec3 sunLight = vec3(1.0, 0.95, 0.88) * (wrapDiffuse * u_sunIntensity);

        // Specular Glints on Oceans and Snowcaps
        vec3 H = normalize(L + V);
        float NdotH = max(0.0, dot(N, H));
        float specIntensity = pow(NdotH, 28.0) * 0.22 * max(0.0, NdotL);
        vec3 specular = vec3(1.0, 0.98, 0.92) * specIntensity;

        vec3 litColor = baseColor * (skyAmbient + sunLight) + specular;

        // --- Atmospheric Rayleigh Scattering & Aerial Perspective ---
        if (u_hazeDensity > 0.001) {
            float distToCam = length(u_camPos - v_worldPos);
            float hazeFactor = 1.0 - exp(-distToCam * 0.0016 * u_hazeDensity);
            hazeFactor = clamp(hazeFactor, 0.0, 0.85);

            // Sun forward-scattering glow in horizon haze
            float VdotL = max(0.0, dot(-V, L));
            vec3 hazeColor = mix(vec3(0.40, 0.56, 0.76), vec3(1.0, 0.82, 0.55), pow(VdotL, 4.0) * 0.65);

            litColor = mix(litColor, hazeColor, hazeFactor);
        }

        // Premultiplied alpha blending
        fragColor = vec4(litColor * alpha, alpha);
    }
    `;

    // Wireframe Grid Shaders
    const gridVS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 a_pos;
    uniform mat4 u_mvp;
    void main() {
        gl_Position = u_mvp * vec4(a_pos, 1.0);
    }
    `;

    const gridFS = `#version 300 es
    precision highp float;
    uniform vec4 u_gridColor;
    out vec4 fragColor;
    void main() {
        fragColor = u_gridColor;
    }
    `;

    // WebGL Pipeline Objects
    let splatProgram, gridProgram;
    let quadVBO, splatCentersVBO, splatScalesVBO, splatRotationsVBO, splatColorsVBO;
    let splatVAO, gridVAO, gridVBO, gridLineCount = 0;

    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    function createProgram(gl, vsSource, fsSource) {
        const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
        const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    const Quat = {
        identity: () => [1, 0, 0, 0], // [w, x, y, z]
        normalize: (q) => {
            const len = Math.hypot(q[0], q[1], q[2], q[3]);
            if (len < 1e-6) return [1, 0, 0, 0];
            return [q[0]/len, q[1]/len, q[2]/len, q[3]/len];
        },
        multiply: (a, b) => [
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3], // w
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2], // x
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1], // y
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]  // z
        ],
        fromAxisAngle: (axis, angle) => {
            const half = angle * 0.5;
            const s = Math.sin(half);
            const len = Math.hypot(axis[0], axis[1], axis[2]) || 1.0;
            return [Math.cos(half), (axis[0]/len)*s, (axis[1]/len)*s, (axis[2]/len)*s];
        },
        toAxes: (q) => {
            const w = q[0], x = q[1], y = q[2], z = q[3];
            // Local Right (+X)
            const right = [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y + w*z),
                2.0 * (x*z - w*y)
            ];
            // Local Up (+Y)
            const up = [
                2.0 * (x*y - w*z),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z + w*x)
            ];
            // Local Forward (-Z in view space)
            const forward = [
                -(2.0 * (x*z + w*y)),
                -(2.0 * (y*z - w*x)),
                -(1.0 - 2.0 * (x*x + y*y))
            ];
            return { right, up, forward };
        },
        toViewMatrix: (pos, q) => {
            const w = q[0], x = q[1], y = q[2], z = q[3];
            // Row 0 = Right (X)
            const r0 = 1.0 - 2.0 * (y*y + z*z);
            const r1 = 2.0 * (x*y + w*z);
            const r2 = 2.0 * (x*z - w*y);

            // Row 1 = Up (Y)
            const u0 = 2.0 * (x*y - w*z);
            const u1 = 1.0 - 2.0 * (x*x + z*z);
            const u2 = 2.0 * (y*z + w*x);

            // Row 2 = Back (-Forward, +Z)
            const b0 = 2.0 * (x*z + w*y);
            const b1 = 2.0 * (y*z - w*x);
            const b2 = 1.0 - 2.0 * (x*x + y*y);

            return [
                r0, u0, b0, 0,
                r1, u1, b1, 0,
                r2, u2, b2, 0,
                -(r0*pos[0] + r1*pos[1] + r2*pos[2]),
                -(u0*pos[0] + u1*pos[1] + u2*pos[2]),
                -(b0*pos[0] + b1*pos[1] + b2*pos[2]),
                1
            ];
        },
        fromLookRotation: (forward, up) => {
            let fLen = Math.hypot(forward[0], forward[1], forward[2]) || 1.0;
            const f = [forward[0]/fLen, forward[1]/fLen, forward[2]/fLen];

            let rx = f[1]*up[2] - f[2]*up[1];
            let ry = f[2]*up[0] - f[0]*up[2];
            let rz = f[0]*up[1] - f[1]*up[0];
            let rLen = Math.hypot(rx, ry, rz);
            if (rLen < 1e-5) {
                rx = 1; ry = 0; rz = 0; rLen = 1;
            }
            const r = [rx/rLen, ry/rLen, rz/rLen];

            const u = [
                r[1]*f[2] - r[2]*f[1],
                r[2]*f[0] - r[0]*f[2],
                r[0]*f[1] - r[1]*f[0]
            ];

            const m00 = r[0], m01 = u[0], m02 = -f[0];
            const m10 = r[1], m11 = u[1], m12 = -f[1];
            const m20 = r[2], m21 = u[2], m22 = -f[2];

            const tr = m00 + m11 + m22;
            let q = [1, 0, 0, 0];
            if (tr > 0) {
                const S = Math.sqrt(tr + 1.0) * 2;
                q = [0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S];
            } else if (m00 > m11 && m00 > m22) {
                const S = Math.sqrt(1.0 + m00 - m11 - m22) * 2;
                q = [(m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S];
            } else if (m11 > m22) {
                const S = Math.sqrt(1.0 + m11 - m00 - m22) * 2;
                q = [(m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S];
            } else {
                const S = Math.sqrt(1.0 + m22 - m00 - m11) * 2;
                q = [(m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S];
            }
            return Quat.normalize(q);
        }
    };

    const Mat4 = {
        identity: () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
        perspective: (fovy, aspect, near, far) => {
            const f = 1.0 / Math.tan(fovy / 2);
            const nf = 1 / (near - far);
            return [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (far + near) * nf, -1,
                0, 0, 2 * far * near * nf, 0
            ];
        },
        lookAt: (eye, target, up) => {
            const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
            let len = Math.hypot(zx, zy, zz);
            const z = len > 0 ? [zx/len, zy/len, zz/len] : [0, 0, 1];

            const xx = up[1]*z[2] - up[2]*z[1];
            const xy = up[2]*z[0] - up[0]*z[2];
            const xz = up[0]*z[1] - up[1]*z[0];
            len = Math.hypot(xx, xy, xz);
            const x = len > 0 ? [xx/len, xy/len, xz/len] : [1, 0, 0];

            const yx = z[1]*x[2] - z[2]*x[1];
            const yy = z[2]*x[0] - z[0]*x[2];
            const yz = z[0]*x[1] - z[1]*x[0];

            return [
                x[0], yx, z[0], 0,
                x[1], yy, z[1], 0,
                x[2], yz, z[2], 0,
                -(x[0]*eye[0] + x[1]*eye[1] + x[2]*eye[2]),
                -(yx*eye[0] + yy*eye[1] + yz*eye[2]),
                -(z[0]*eye[0] + z[1]*eye[1] + z[2]*eye[2]),
                1
            ];
        },
        multiply: (a, b) => {
            const out = new Array(16);
            for (let r = 0; r < 4; r++) {
                for (let c = 0; c < 4; c++) {
                    out[r*4 + c] = a[r*4 + 0]*b[0*4 + c] +
                                   a[r*4 + 1]*b[1*4 + c] +
                                   a[r*4 + 2]*b[2*4 + c] +
                                   a[r*4 + 3]*b[3*4 + c];
                }
            }
            return out;
        },
        extractFrustumPlanes: (m) => {
            // Returns 6 planes [A, B, C, D] where Ax + By + Cz + D >= 0 is inside
            const planes = [];
            // Left
            planes.push([m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]]);
            // Right
            planes.push([m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]]);
            // Bottom
            planes.push([m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]]);
            // Top
            planes.push([m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]]);
            // Near
            planes.push([m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]]);
            // Far
            planes.push([m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]]);

            // Normalize planes
            for (let i = 0; i < 6; i++) {
                const p = planes[i];
                const len = Math.hypot(p[0], p[1], p[2]);
                if (len > 0) {
                    p[0] /= len; p[1] /= len; p[2] /= len; p[3] /= len;
                }
            }
            return planes;
        }
    };

    function initGL() {
        splatProgram = createProgram(gl, splatVS, splatFS);
        gridProgram = createProgram(gl, gridVS, gridFS);

        const quadData = new Float32Array([
            -2.0, -2.0,
             2.0, -2.0,
            -2.0,  2.0,
            -2.0,  2.0,
             2.0, -2.0,
             2.0,  2.0
        ]);

        splatVAO = gl.createVertexArray();
        gl.bindVertexArray(splatVAO);

        quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, quadData, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        // Dynamic Instanced Buffers for Visible Splats
        splatCentersVBO = gl.createBuffer();
        splatScalesVBO = gl.createBuffer();
        splatRotationsVBO = gl.createBuffer();
        splatColorsVBO = gl.createBuffer();

        gl.bindBuffer(gl.ARRAY_BUFFER, splatCentersVBO);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(1, 1);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatScalesVBO);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(2, 1);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatRotationsVBO);
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(3, 1);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatColorsVBO);
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 4, gl.UNSIGNED_BYTE, true, 0, 0);
        gl.vertexAttribDivisor(4, 1);

        gl.bindVertexArray(null);

        initSpheroidGrid();
    }

    function initSpheroidGrid() {
        const lines = [];
        const a = state.spheroid.radius || 100.0;
        const b = a * (1.0 - (state.spheroid.flattening || 0.05));

        const latSteps = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75];
        const lonSegments = 96;

        for (const latDeg of latSteps) {
            const phi = latDeg * Math.PI / 180;
            const cosPhi = Math.cos(phi);
            const sinPhi = Math.sin(phi);

            for (let i = 0; i <= lonSegments; i++) {
                const theta = (i / lonSegments) * 2 * Math.PI;
                const x = a * cosPhi * Math.cos(theta);
                const y = a * cosPhi * Math.sin(theta);
                const z = b * sinPhi;

                lines.push(x, y, z);
                if (i > 0 && i < lonSegments) {
                    lines.push(x, y, z);
                }
            }
        }

        const lonSteps = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150];
        const latSegments = 48;
        for (const lonDeg of lonSteps) {
            const theta = lonDeg * Math.PI / 180;
            for (let i = 0; i <= latSegments; i++) {
                const phi = (-90 + (i / latSegments) * 180) * Math.PI / 180;
                const x = a * Math.cos(phi) * Math.cos(theta);
                const y = a * Math.cos(phi) * Math.sin(theta);
                const z = b * Math.sin(phi);

                lines.push(x, y, z);
                if (i > 0 && i < latSegments) {
                    lines.push(x, y, z);
                }
            }
        }

        gridLineCount = lines.length / 3;

        gridVAO = gl.createVertexArray();
        gl.bindVertexArray(gridVAO);
        gridVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, gridVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    function framePlanet() {
        const R = state.spheroid.radius || 100.0;
        const dist = R * 2.4;

        state.camera.orbitDistance = dist;
        state.camera.orbitTheta = 0;
        state.camera.orbitPhi = 0.45;

        if (state.cameraMode === 'fly') {
            state.camera.pos = [0, -dist * 0.85, dist * 0.5];
            const lookDir = [-state.camera.pos[0], -state.camera.pos[1], -state.camera.pos[2]];
            state.camera.quat = Quat.fromLookRotation(lookDir, [0, 0, 1]);
        }

        state.flightSpeed = 50.0;
        const sliderSpeed = document.getElementById('slider-speed');
        if (sliderSpeed) {
            sliderSpeed.value = state.flightSpeed;
            document.getElementById('val-speed').innerText = state.flightSpeed.toFixed(0) + ' km/s';
        }
    }

    function formatLargeNumber(n) {
        if (n >= 1e15) return (n / 1e15).toFixed(2) + ' P';
        if (n >= 1e12) return (n / 1e12).toFixed(2) + ' T';
        if (n >= 1e9)  return (n / 1e9).toFixed(2) + ' B';
        if (n >= 1e6)  return (n / 1e6).toFixed(2) + ' M';
        if (n >= 1e3)  return (n / 1e3).toFixed(1) + ' k';
        return n.toLocaleString();
    }

    function formatByteSize(bytes) {
        if (bytes >= 1e15) return (bytes / (1024**5)).toFixed(2) + ' PB';
        if (bytes >= 1e12) return (bytes / (1024**4)).toFixed(2) + ' TB';
        if (bytes >= 1e9)  return (bytes / (1024**3)).toFixed(2) + ' GB';
        if (bytes >= 1e6)  return (bytes / (1024**2)).toFixed(2) + ' MB';
        if (bytes >= 1e3)  return (bytes / 1024).toFixed(1) + ' KB';
        return bytes + ' B';
    }

    // Load Chunk Manifests & Binary Splat Data
    async function fetchSplatData() {
        showLoader('Streaming chunk manifests & LODs...');
        try {
            const metaRes = await fetch('/api/meta');
            const meta = await metaRes.json();
            const prevRadius = state.spheroid.radius;

            if (meta.spheroid && meta.spheroid.radius > 0) {
                state.spheroid = meta.spheroid;
                initSpheroidGrid();
                document.getElementById('gen-radius').value = state.spheroid.radius;
                document.getElementById('gen-flattening').value = state.spheroid.flattening;

                const currentDist = Math.hypot(state.camera.pos[0], state.camera.pos[1], state.camera.pos[2]);
                if (currentDist < state.spheroid.radius * 1.05 || Math.abs(state.spheroid.radius - prevRadius) > 10) {
                    framePlanet();
                }
            }

            if (meta.density && meta.density.meanSpacingMeters) {
                const sp = meta.density.meanSpacingMeters;
                state.meanSpacingKm = sp / 1000.0;
                document.getElementById('metric-spacing').innerText = sp >= 1000 ? (sp / 1000).toFixed(2) + ' km' : sp.toFixed(1) + ' m';
            } else {
                const totalPts = meta.splatCount || 100000;
                const surfArea = 4.0 * Math.PI * (state.spheroid.radius || 100.0) * (state.spheroid.radius || 100.0);
                state.meanSpacingKm = Math.sqrt(surfArea / totalPts);
            }

            // Fetch Chunk Manifests
            const chunksRes = await fetch('/api/chunks');
            const chunksData = await chunksRes.json();
            state.chunkManifests = chunksData.chunks || [];
            state.splatCount = meta.splatCount || 0;
            state.virtualPoints = meta.virtualPoints || meta.splatCount || 0;
            state.activeSplatCount = 0;

            if (meta.virtualPoints && meta.virtualPoints > state.splatCount) {
                document.getElementById('metric-splats').innerText = `${formatLargeNumber(meta.virtualPoints)} (${formatLargeNumber(state.splatCount)} grid)`;
            } else {
                document.getElementById('metric-splats').innerText = formatLargeNumber(state.splatCount);
            }
            document.getElementById('metric-bytes').innerText = formatByteSize(state.splatCount * 32);

            // Populate Chunks List (without downloading all chunks at once)
            await initChunksData();

            // Trigger immediate initial visible chunk stream (only loads visible chunks at appropriate LOD)
            const aspect = canvas.width / canvas.height;
            const near = 0.01;
            const far = (state.spheroid.radius || 100) * 20.0;
            const proj = Mat4.perspective(state.camera.fov, aspect, near, far);
            const view = Mat4.lookAt(state.camera.pos, state.camera.target, [0, 0, 1]);
            await updateChunkCullingAndLOD(view, proj, aspect, near, far);

            hideLoader();
        } catch (err) {
            console.error('Error fetching splats:', err);
            showLoader('Error loading splats: ' + err.message);
        }
    }

    async function initChunksData() {
        state.chunks = [];
        const manifests = state.chunkManifests;

        for (let i = 0; i < manifests.length; i++) {
            const m = manifests[i];
            state.chunks.push({
                id: m.id,
                center: m.center,
                radius: m.radius,
                normal: m.normal,
                countsLOD: m.countsLOD,
                lods: [null, null, null, null, null], // Cached TypedArrays for LOD 0, 1, 2, 3, 4
                activeLOD: 4,
                currentLOD: undefined,
                prevLOD: null,
                fadeStart: 0,
                lastUsed: 0,
                isVisible: true
            });
        }
    }

    const MAX_CACHED_CHUNKS = 96; // Max cached in-memory chunk LOD arrays to keep browser RAM low
    const inflightRequests = new Map();

    function evictStaleChunks() {
        let cachedCount = 0;
        const cached = [];
        for (let i = 0; i < state.chunks.length; i++) {
            const ch = state.chunks[i];
            let hasData = false;
            for (let l = 0; l < 5; l++) {
                if (ch.lods[l]) hasData = true;
            }
            if (hasData) {
                cachedCount++;
                if (!ch.isVisible) {
                    cached.push(ch);
                }
            }
        }

        const maxAllowed = Math.max(MAX_CACHED_CHUNKS, Math.min(256, Math.floor(state.chunks.length * 0.3)));
        if (cachedCount > maxAllowed) {
            cached.sort((a, b) => (a.lastUsed || 0) - (b.lastUsed || 0));
            const toEvict = cachedCount - maxAllowed;
            for (let i = 0; i < toEvict && i < cached.length; i++) {
                cached[i].lods = [null, null, null, null, null];
            }
        }
    }

    async function loadChunkLOD(chunkId, lod) {
        const chunk = state.chunks[chunkId];
        if (!chunk) return null;
        if (chunk.lods[lod]) {
            chunk.lastUsed = performance.now();
            return chunk.lods[lod];
        }

        const key = `${chunkId}_${lod}`;
        if (inflightRequests.has(key)) {
            return inflightRequests.get(key);
        }

        const reqPromise = (async () => {
            try {
                const res = await fetch(`/api/chunk?id=${chunkId}&lod=${lod}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const buf = await res.arrayBuffer();
                const count = Math.floor(buf.byteLength / 32);

                const centers = new Float32Array(count * 3);
                const scales = new Float32Array(count * 3);
                const rotations = new Float32Array(count * 4);
                const colors = new Uint8Array(count * 4);
                const view = new DataView(buf);

                for (let i = 0; i < count; i++) {
                    const offset = i * 32;
                    centers[i*3+0] = view.getFloat32(offset + 0, true);
                    centers[i*3+1] = view.getFloat32(offset + 4, true);
                    centers[i*3+2] = view.getFloat32(offset + 8, true);

                    scales[i*3+0] = view.getFloat32(offset + 12, true);
                    scales[i*3+1] = view.getFloat32(offset + 16, true);
                    scales[i*3+2] = view.getFloat32(offset + 20, true);

                    colors[i*4+0] = view.getUint8(offset + 24);
                    colors[i*4+1] = view.getUint8(offset + 25);
                    colors[i*4+2] = view.getUint8(offset + 26);
                    colors[i*4+3] = view.getUint8(offset + 27);

                    for (let j = 0; j < 4; j++) {
                        rotations[i*4+j] = (view.getUint8(offset + 28 + j) - 128.0) / 128.0;
                    }
                }

                chunk.lastUsed = performance.now();
                chunk.lods[lod] = { count, centers, scales, rotations, colors };
                return chunk.lods[lod];
            } catch (e) {
                console.warn(`Failed to stream chunk ${chunkId} LOD ${lod}:`, e);
                return null;
            } finally {
                inflightRequests.delete(key);
            }
        })();

        inflightRequests.set(key, reqPromise);
        return reqPromise;
    }

    function parseSplatBuffer(buffer) {
        const count = Math.floor(buffer.byteLength / 32);
        state.splatCount = count;
        state.activeSplatCount = count;

        const centers = new Float32Array(count * 3);
        const scales = new Float32Array(count * 3);
        const rotations = new Float32Array(count * 4);
        const colors = new Uint8Array(count * 4);

        const dataView = new DataView(buffer);

        for (let i = 0; i < count; i++) {
            const offset = i * 32;
            centers[i*3 + 0] = dataView.getFloat32(offset + 0, true);
            centers[i*3 + 1] = dataView.getFloat32(offset + 4, true);
            centers[i*3 + 2] = dataView.getFloat32(offset + 8, true);

            scales[i*3 + 0] = dataView.getFloat32(offset + 12, true);
            scales[i*3 + 1] = dataView.getFloat32(offset + 16, true);
            scales[i*3 + 2] = dataView.getFloat32(offset + 20, true);

            colors[i*4 + 0] = dataView.getUint8(offset + 24);
            colors[i*4 + 1] = dataView.getUint8(offset + 25);
            colors[i*4 + 2] = dataView.getUint8(offset + 26);
            colors[i*4 + 3] = dataView.getUint8(offset + 27);

            for (let j = 0; j < 4; j++) {
                rotations[i*4 + j] = (dataView.getUint8(offset + 28 + j) - 128.0) / 128.0;
            }
        }

        state.splats = {
            count,
            centers,
            scales,
            rotations,
            colors
        };

        // Upload initial full buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, splatCentersVBO);
        gl.bufferData(gl.ARRAY_BUFFER, centers, gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatScalesVBO);
        gl.bufferData(gl.ARRAY_BUFFER, scales, gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatRotationsVBO);
        gl.bufferData(gl.ARRAY_BUFFER, rotations, gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, splatColorsVBO);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);

        document.getElementById('metric-splats').innerText = count.toLocaleString();
        document.getElementById('metric-bytes').innerText = (buffer.byteLength / (1024 * 1024)).toFixed(2) + ' MB';
    }

    // Spatial Chunk Culling & LOD Evaluation
    async function updateChunkCullingAndLOD(view, proj, aspect, near, far) {
        if (!state.chunks || state.chunks.length === 0) return;
        if (state.isUpdatingLOD) return;
        state.isUpdatingLOD = true;

        try {
            const camPos = state.camera.pos;
            const R = state.spheroid.radius || 100.0;
            const bias = state.lodBias || 1.0;
            const camDist = Math.hypot(camPos[0], camPos[1], camPos[2]);
            const alt = state.altitude !== undefined ? state.altitude : Math.max(0.0001, camDist - R);
            const tanHalfFov = Math.tan(state.camera.fov / 2.0);

            // 1. Geometric Line-of-Sight Horizon Range
            // The maximum Euclidean distance from camera to any point on the visible spherical cap:
            // d_horizon = sqrt(2 * R * (alt + elevMax) + (alt + elevMax)^2) + sqrt(2 * R * elevMax)
            const elevMax = (state.spheroid.radius || 100.0) * 0.15; // Mountain peaks elevation safety margin
            const effAlt = alt + elevMax;
            const dHorizon = Math.sqrt(Math.max(0.0, 2.0 * R * effAlt + effAlt * effAlt)) + Math.sqrt(Math.max(0.0, 2.0 * R * elevMax));
            const maxVisibleDist = dHorizon + Math.max(15.0, R * 0.08);

            // Planetary horizon angle from planet center
            const cosHoriz = Math.min(1.0, Math.max(0.0, R / camDist));
            const horizonAngle = Math.acos(cosHoriz);

            let visibleChunks = 0;
            const visibleList = [];

            for (let i = 0; i < state.chunks.length; i++) {
                const ch = state.chunks[i];
                const cX = ch.center[0], cY = ch.center[1], cZ = ch.center[2];
                const radius = ch.radius;

                // 2. Strict Horizon Line-of-Sight Distance Culling
                // If a chunk is physically behind the horizon curvature, cull it immediately
                const distToCam = Math.hypot(cX - camPos[0], cY - camPos[1], cZ - camPos[2]);
                if (distToCam - radius > maxVisibleDist) {
                    ch.isVisible = false;
                    continue;
                }

                // 3. Angular Planetary Horizon Occlusion Culling
                const chunkDist = Math.hypot(cX, cY, cZ) || 1.0;
                const dotCamChunk = (camPos[0]*cX + camPos[1]*cY + camPos[2]*cZ) / (camDist * chunkDist);
                const chunkAngularRadius = radius / R;
                const minCos = Math.cos(Math.min(Math.PI, horizonAngle + chunkAngularRadius + 0.04));

                if (camDist > R && dotCamChunk < minCos) {
                    ch.isVisible = false;
                    continue;
                }

                // 4. View-Space Frustum Culling
                const vX = view[0]*cX + view[4]*cY + view[8]*cZ + view[12];
                const vY = view[1]*cX + view[5]*cY + view[9]*cZ + view[13];
                const vZ = view[2]*cX + view[6]*cY + view[10]*cZ + view[14];

                const viewDepth = -vZ;
                const effectiveFar = Math.min(far, maxVisibleDist + radius);
                if (viewDepth < near - radius || viewDepth > effectiveFar + radius) {
                    ch.isVisible = false;
                    continue;
                }

                const halfH = viewDepth * tanHalfFov + radius * Math.sqrt(1.0 + tanHalfFov * tanHalfFov);
                const halfW = viewDepth * (tanHalfFov * aspect) + radius * Math.sqrt(1.0 + (tanHalfFov * aspect) * (tanHalfFov * aspect));

                if (Math.abs(vY) > halfH || Math.abs(vX) > halfW) {
                    ch.isVisible = false;
                    continue;
                }

                ch.isVisible = true;
                visibleChunks++;
                visibleList.push(ch);

                // 5. Angular Screen-Space Splat Pixel Footprint LOD Selection
                // Measure exact 3D distance to closest terrain chunk boundary/summit rather than distant sea-level ellipsoid
                const closestDist = Math.max(0.001, distToCam - radius * 0.65);
                const focalPx = (canvas.height * 0.5) / tanHalfFov;
                const nominalSpacing = state.meanSpacingKm || 1.0;
                const projectedSplatPixels = (nominalSpacing / closestDist) * focalPx * bias;

                let targetLOD = 0;
                if (projectedSplatPixels >= 2.2) {
                    // Full Resolution (LOD 0, stride 1) - Splats occupy >= 2.2 screen pixels
                    targetLOD = 0;
                } else if (projectedSplatPixels >= 0.9) {
                    // Medium Resolution (LOD 1, stride 2) - 4x reduction
                    targetLOD = 1;
                } else if (projectedSplatPixels >= 0.35) {
                    // Coarse Resolution (LOD 2, stride 4) - 16x reduction
                    targetLOD = 2;
                } else if (projectedSplatPixels >= 0.12) {
                    // Orbit Resolution (LOD 3, stride 8) - 64x reduction
                    targetLOD = 3;
                } else {
                    // Deep Orbit Resolution (LOD 4, stride 16) - 256x reduction!
                    targetLOD = 4;
                }

                // Smooth crossfade LOD management
                if (ch.currentLOD === undefined) {
                    ch.currentLOD = targetLOD;
                    ch.prevLOD = null;
                    ch.fadeStart = 0;
                } else if (ch.currentLOD !== targetLOD && ch.prevLOD === null) {
                    ch.prevLOD = ch.currentLOD;
                    ch.currentLOD = targetLOD;
                    ch.fadeStart = performance.now();
                    state.hasActiveLODFade = true;
                }
            }

            state.activeChunksCount = visibleChunks;
            document.getElementById('metric-chunks').innerText = `${visibleChunks} / ${state.chunks.length}`;

            // Evict distant non-visible chunks from client cache to keep RAM footprint low
            evictStaleChunks();

            // Preload any missing chunk LODs in parallel
            const fetchPromises = [];
            for (const ch of visibleList) {
                if (!ch.lods[ch.currentLOD]) {
                    fetchPromises.push(loadChunkLOD(ch.id, ch.currentLOD));
                }
                if (ch.prevLOD !== null && ch.prevLOD !== undefined && !ch.lods[ch.prevLOD]) {
                    fetchPromises.push(loadChunkLOD(ch.id, ch.prevLOD));
                }
            }
            if (fetchPromises.length > 0) {
                await Promise.all(fetchPromises);
            }

            // Assemble active splat buffers with smooth LOD alpha crossfading
            if (state.chunks.length > 0 && visibleChunks > 0) {
                const now = performance.now();
                const FADE_DURATION = 350; // 350ms seamless crossfade
                let hasOngoingFade = false;

                const loadedEntries = [];
                let totalActive = 0;

                for (const ch of visibleList) {
                    let currWeight = 1.0;
                    let prevWeight = 0.0;

                    if (ch.prevLOD !== null && ch.prevLOD !== undefined) {
                        const elapsed = now - ch.fadeStart;
                        const progress = Math.min(1.0, elapsed / FADE_DURATION);
                        if (progress >= 1.0) {
                            ch.prevLOD = null;
                        } else {
                            hasOngoingFade = true;
                            // Smooth Hermite blend curve: 3t^2 - 2t^3
                            const t = progress * progress * (3.0 - 2.0 * progress);
                            currWeight = t;
                            prevWeight = 1.0 - t;
                        }
                    }

                    // Current LOD splats
                    const currData = ch.lods[ch.currentLOD];
                    if (currData && currWeight > 0.01) {
                        loadedEntries.push({ data: currData, weight: currWeight, id: ch.id, lod: ch.currentLOD });
                        totalActive += currData.count;
                    }

                    // Previous LOD splats during crossfade
                    if (ch.prevLOD !== null && prevWeight > 0.01) {
                        const prevData = ch.lods[ch.prevLOD];
                        if (prevData) {
                            loadedEntries.push({ data: prevData, weight: prevWeight, id: ch.id, lod: ch.prevLOD });
                            totalActive += prevData.count;
                        }
                    }
                }

                state.hasActiveLODFade = hasOngoingFade;

                // Check signature to avoid redundant GPU VBO re-allocations when scene is static
                const sig = loadedEntries.map(e => `${e.id}:${e.lod}:${hasOngoingFade ? e.weight.toFixed(2) : '1'}`).join(',');
                if (sig !== state.lastBufferSig || hasOngoingFade) {
                    state.lastBufferSig = sig;

                    if (totalActive > 0) {
                        const mergedCenters = new Float32Array(totalActive * 3);
                        const mergedScales = new Float32Array(totalActive * 3);
                        const mergedRotations = new Float32Array(totalActive * 4);
                        const mergedColors = new Uint8Array(totalActive * 4);

                        let offset = 0;
                        for (const entry of loadedEntries) {
                            const d = entry.data;
                            const w = entry.weight;

                            mergedCenters.set(d.centers, offset * 3);
                            mergedScales.set(d.scales, offset * 3);
                            mergedRotations.set(d.rotations, offset * 4);

                            if (w >= 0.99) {
                                mergedColors.set(d.colors, offset * 4);
                            } else {
                                // Apply crossfade transparency weight to alpha channel
                                const colSlice = new Uint8Array(d.colors.length);
                                colSlice.set(d.colors);
                                for (let k = 0; k < d.count; k++) {
                                    colSlice[k * 4 + 3] = Math.round(d.colors[k * 4 + 3] * w);
                                }
                                mergedColors.set(colSlice, offset * 4);
                            }
                            offset += d.count;
                        }

                        state.activeSplatCount = totalActive;

                        gl.bindBuffer(gl.ARRAY_BUFFER, splatCentersVBO);
                        gl.bufferData(gl.ARRAY_BUFFER, mergedCenters, gl.DYNAMIC_DRAW);

                        gl.bindBuffer(gl.ARRAY_BUFFER, splatScalesVBO);
                        gl.bufferData(gl.ARRAY_BUFFER, mergedScales, gl.DYNAMIC_DRAW);

                        gl.bindBuffer(gl.ARRAY_BUFFER, splatRotationsVBO);
                        gl.bufferData(gl.ARRAY_BUFFER, mergedRotations, gl.DYNAMIC_DRAW);

                        gl.bindBuffer(gl.ARRAY_BUFFER, splatColorsVBO);
                        gl.bufferData(gl.ARRAY_BUFFER, mergedColors, gl.DYNAMIC_DRAW);

                        document.getElementById('metric-splats').innerText = `${formatLargeNumber(totalActive)} / ${formatLargeNumber(state.splatCount)}`;
                    }
                }
            } else {
                state.activeSplatCount = 0;
                document.getElementById('metric-splats').innerText = `0 / ${formatLargeNumber(state.splatCount)}`;
            }
        } finally {
            state.isUpdatingLOD = false;
        }
    }

    // Calculate exact oblate spheroid surface radius at a given 3D position
    function getLocalSurfaceRadius(pos, a, f) {
        const b = a * (1.0 - f);
        const r = Math.hypot(pos[0], pos[1], pos[2]) || 1.0;
        const p = Math.hypot(pos[0], pos[1]);
        const cosLat = p / r;
        const sinLat = pos[2] / r;
        const denom = Math.sqrt((b * cosLat) * (b * cosLat) + (a * sinLat) * (a * sinLat));
        return denom > 1e-6 ? (a * b) / denom : b;
    }

    // Camera Navigation & Physics Update (Logarithmic Adaptive Speed & 6-DOF Quaternion Controls)
    function updateCamera(dt) {
        const camDist = Math.hypot(state.camera.pos[0], state.camera.pos[1], state.camera.pos[2]);
        const localR = getLocalSurfaceRadius(state.camera.pos, state.spheroid.radius, state.spheroid.flattening);
        const alt = Math.max(0.0001, camDist - localR);
        state.altitude = alt;

        if (state.cameraMode === 'fly') {
            const cam = state.camera;
            const axes = Quat.toAxes(cam.quat);
            const forward = axes.forward;
            const right = axes.right;
            const up = axes.up;

            // 6-DOF Roll controls with Q / E keys (no gimbal lock!)
            const rollSpeed = 1.6; // rad/s
            let roll = 0;
            if (state.keys['KeyQ']) roll += rollSpeed * dt;
            if (state.keys['KeyE']) roll -= rollSpeed * dt;
            if (roll !== 0) {
                const qRoll = Quat.fromAxisAngle(forward, roll);
                cam.quat = Quat.normalize(Quat.multiply(qRoll, cam.quat));
            }

            // Smooth logarithmic speed: fast in high orbit, ultra-smooth and precise at 1km/low altitude
            const speedBoost = (state.keys['ShiftLeft'] || state.keys['ShiftRight']) ? 5.0 : 1.0;
            const slowMode = (state.keys['AltLeft'] || state.keys['ControlLeft']) ? 0.15 : 1.0;

            const adaptiveFactor = Math.max(0.005, (alt / 15.0) * 0.4 + 0.015);
            const speed = (state.flightSpeed * adaptiveFactor * speedBoost * slowMode) * dt;

            let moveFwd = 0;
            let moveRight = 0;
            let moveUp = 0;

            if (state.keys['KeyW']) moveFwd += 1;
            if (state.keys['KeyS']) moveFwd -= 1;
            if (state.keys['KeyA']) moveRight -= 1;
            if (state.keys['KeyD']) moveRight += 1;
            if (state.keys['Space']) moveUp += 1;
            if (state.keys['KeyC']) moveUp -= 1;

            cam.pos[0] += forward[0] * moveFwd * speed + right[0] * moveRight * speed + up[0] * moveUp * speed;
            cam.pos[1] += forward[1] * moveFwd * speed + right[1] * moveRight * speed + up[1] * moveUp * speed;
            cam.pos[2] += forward[2] * moveFwd * speed + right[2] * moveRight * speed + up[2] * moveUp * speed;

            // Surface collision floor: enforces 0.5 km minimum altitude above surface
            const minFlightAlt = 0.5;
            const minAllowedDist = localR + minFlightAlt;
            const newDist = Math.hypot(cam.pos[0], cam.pos[1], cam.pos[2]);
            if (newDist < minAllowedDist) {
                const pushOut = minAllowedDist / newDist;
                cam.pos[0] *= pushOut;
                cam.pos[1] *= pushOut;
                cam.pos[2] *= pushOut;
            }

            cam.target = [
                cam.pos[0] + forward[0],
                cam.pos[1] + forward[1],
                cam.pos[2] + forward[2]
            ];
        } else {
            // Orbit Mode
            const cam = state.camera;
            const x = cam.orbitDistance * Math.cos(cam.orbitPhi) * Math.cos(cam.orbitTheta);
            const y = cam.orbitDistance * Math.cos(cam.orbitPhi) * Math.sin(cam.orbitTheta);
            const z = cam.orbitDistance * Math.sin(cam.orbitPhi);

            cam.pos = [x, y, z];
            cam.target = [0, 0, 0];
        }

        updateTelemetry();
    }

    function updateTelemetry() {
        const pos = state.camera.pos;
        const a = state.spheroid.radius || 100.0;
        const f = state.spheroid.flattening || 0.05;
        const b = a * (1.0 - f);
        const e2 = (a*a - b*b) / (a*a);

        const p = Math.hypot(pos[0], pos[1]);
        const lonDeg = Math.atan2(pos[1], pos[0]) * 180 / Math.PI;

        let latDeg = 0;
        let alt = 0;
        if (p < 1e-4) {
            latDeg = pos[2] > 0 ? 90 : -90;
            alt = Math.abs(pos[2]) - b;
        } else {
            const latRad = Math.atan2(pos[2], p * (1.0 - e2));
            latDeg = latRad * 180 / Math.PI;
            const sinLat = Math.sin(latRad);
            const N = a / Math.sqrt(1.0 - e2 * sinLat * sinLat);
            alt = p / Math.cos(latRad) - N;
        }

        document.getElementById('tele-lat').innerText = latDeg.toFixed(2) + '°';
        document.getElementById('tele-lon').innerText = lonDeg.toFixed(2) + '°';

        if (alt < 1.0) {
            document.getElementById('tele-alt').innerText = (alt * 1000.0).toFixed(1) + ' m';
        } else if (alt < 1000.0) {
            document.getElementById('tele-alt').innerText = alt.toFixed(2) + ' km';
        } else {
            document.getElementById('tele-alt').innerText = alt.toLocaleString(undefined, {maximumFractionDigits: 1}) + ' km';
        }

        document.getElementById('tele-cam-mode').innerText = state.cameraMode.toUpperCase();
    }

    // Render Frame
    function render(time) {
        requestAnimationFrame(render);

        const dt = Math.min((time - state.lastTime) / 1000, 0.1);
        state.lastTime = time;

        state.frameCount++;
        if (time - state.fpsTimer >= 500) {
            state.fps = Math.round((state.frameCount * 1000) / (time - state.fpsTimer));
            state.frameCount = 0;
            state.fpsTimer = time;
            document.getElementById('metric-fps').innerText = state.fps;
        }

        if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            gl.viewport(0, 0, canvas.width, canvas.height);
        }

        updateCamera(dt);

        gl.clearColor(0.02, 0.04, 0.07, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const aspect = canvas.width / canvas.height;
        const R = state.spheroid.radius || 100.0;
        const camDist = Math.hypot(state.camera.pos[0], state.camera.pos[1], state.camera.pos[2]);
        const alt = Math.max(0.0001, camDist - R);

        // Near plane scales from centimeters up to kilometers without clipping close terrain
        const near = Math.max(0.00001, Math.min(0.2, alt * 0.002));
        const far = Math.max(100000.0, (R + camDist) * 10.0);

        const proj = Mat4.perspective(state.camera.fov, aspect, near, far);
        let view;
        if (state.cameraMode === 'fly') {
            view = Quat.toViewMatrix(state.camera.pos, state.camera.quat);
        } else {
            view = Mat4.lookAt(state.camera.pos, state.camera.target, [0, 0, 1]);
        }
        const mvp = Mat4.multiply(proj, view);

        // Periodic Chunk Culling & Continuous LOD Crossfade update
        if (state.hasActiveLODFade || time - state.lastLODUpdate > 100) {
            updateChunkCullingAndLOD(view, proj, aspect, near, far);
            state.lastLODUpdate = time;
        }

        // Render Spheroid Grid
        if (state.showGrid && gridLineCount > 0) {
            gl.useProgram(gridProgram);
            gl.uniformMatrix4fv(gl.getUniformLocation(gridProgram, 'u_mvp'), false, new Float32Array(mvp));
            gl.uniform4f(gl.getUniformLocation(gridProgram, 'u_gridColor'), 0.0, 0.6, 1.0, 0.25);

            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            gl.disable(gl.DEPTH_TEST);

            gl.bindVertexArray(gridVAO);
            gl.drawArrays(gl.LINES, 0, gridLineCount);
            gl.bindVertexArray(null);
        }

        // Render Active Splats
        const drawCount = state.activeSplatCount || 0;
        if (drawCount > 0) {
            gl.useProgram(splatProgram);

            const fy = 0.5 * canvas.height / Math.tan(state.camera.fov / 2);
            const fx = fy;

            gl.uniformMatrix4fv(gl.getUniformLocation(splatProgram, 'u_view'), false, new Float32Array(view));
            gl.uniformMatrix4fv(gl.getUniformLocation(splatProgram, 'u_projection'), false, new Float32Array(proj));
            gl.uniform3f(gl.getUniformLocation(splatProgram, 'u_camPos'), state.camera.pos[0], state.camera.pos[1], state.camera.pos[2]);
            gl.uniform2f(gl.getUniformLocation(splatProgram, 'u_focal'), fx, fy);
            gl.uniform2f(gl.getUniformLocation(splatProgram, 'u_viewport'), canvas.width, canvas.height);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_scaleMult'), state.scaleMultiplier);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_thicknessMult'), state.thicknessMultiplier);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_alphaBoost'), state.alphaBoost);
            gl.uniform1i(gl.getUniformLocation(splatProgram, 'u_colorMode'), state.colorMode);
            gl.uniform1i(gl.getUniformLocation(splatProgram, 'u_kernelType'), state.kernelType || 0);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_spheroidRadius'), state.spheroid.radius);

            // Sun Lighting & Atmosphere Uniforms
            const azRad = (state.sunAzimuth || 45.0) * Math.PI / 180.0;
            const elRad = (state.sunElevation || 35.0) * Math.PI / 180.0;
            const sunDirX = Math.cos(elRad) * Math.cos(azRad);
            const sunDirY = Math.cos(elRad) * Math.sin(azRad);
            const sunDirZ = Math.sin(elRad);

            gl.uniform3f(gl.getUniformLocation(splatProgram, 'u_sunDir'), sunDirX, sunDirY, sunDirZ);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_sunIntensity'), state.sunIntensity || 1.25);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_ambientIntensity'), state.ambientIntensity || 0.38);
            gl.uniform1f(gl.getUniformLocation(splatProgram, 'u_hazeDensity'), state.hazeDensity || 0.50);
            gl.uniform1i(gl.getUniformLocation(splatProgram, 'u_shadingEnabled'), state.shadingEnabled ? 1 : 0);

            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.disable(gl.DEPTH_TEST);

            gl.bindVertexArray(splatVAO);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, drawCount);
            gl.bindVertexArray(null);
        }
    }

    function setupUI() {
        window.addEventListener('keydown', (e) => {
            state.keys[e.code] = true;
        });
        window.addEventListener('keyup', (e) => {
            state.keys[e.code] = false;
        });

        canvas.addEventListener('mousedown', (e) => {
            state.isDragging = true;
            state.lastMouseX = e.clientX;
            state.lastMouseY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            state.isDragging = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!state.isDragging) return;

            const dx = e.clientX - state.lastMouseX;
            const dy = e.clientY - state.lastMouseY;
            state.lastMouseX = e.clientX;
            state.lastMouseY = e.clientY;

            if (state.cameraMode === 'fly') {
                const sensitivity = 0.0025;
                const axes = Quat.toAxes(state.camera.quat);
                // Pitch (around local right axis - zero gimbal lock!)
                const qPitch = Quat.fromAxisAngle(axes.right, -dy * sensitivity);
                // Yaw (around local up axis)
                const qYaw = Quat.fromAxisAngle(axes.up, -dx * sensitivity);
                const qRot = Quat.multiply(qYaw, qPitch);
                state.camera.quat = Quat.normalize(Quat.multiply(qRot, state.camera.quat));
            } else {
                const sensitivity = 0.005;
                state.camera.orbitTheta -= dx * sensitivity;
                state.camera.orbitPhi += dy * sensitivity;
                state.camera.orbitPhi = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.48, state.camera.orbitPhi));
            }
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const R = state.spheroid.radius || 100.0;
            if (state.cameraMode === 'orbit') {
                const step = Math.max(0.1, state.camera.orbitDistance * 0.05);
                state.camera.orbitDistance = Math.max(R * 1.001, state.camera.orbitDistance + Math.sign(e.deltaY) * step);
            } else {
                state.flightSpeed = Math.max(1, Math.min(500, state.flightSpeed - Math.sign(e.deltaY) * 5));
                document.getElementById('slider-speed').value = state.flightSpeed;
                document.getElementById('val-speed').innerText = state.flightSpeed.toFixed(0) + ' km/s';
            }
        }, { passive: false });

        const sidebar = document.getElementById('sidebar');
        const openBtn = document.getElementById('open-sidebar-btn');
        const closeBtn = document.getElementById('toggle-sidebar-btn');

        closeBtn.addEventListener('click', () => {
            sidebar.classList.add('closed');
            openBtn.classList.remove('hidden');
        });

        openBtn.addEventListener('click', () => {
            sidebar.classList.remove('closed');
            openBtn.classList.add('hidden');
        });

        const btnFly = document.getElementById('btn-mode-fly');
        const btnOrbit = document.getElementById('btn-mode-orbit');
        const flyHelp = document.getElementById('fly-help');
        const orbitHelp = document.getElementById('orbit-help');

        btnFly.addEventListener('click', () => {
            if (state.cameraMode !== 'fly') {
                state.cameraMode = 'fly';
                const fwd = [
                    state.camera.target[0] - state.camera.pos[0],
                    state.camera.target[1] - state.camera.pos[1],
                    state.camera.target[2] - state.camera.pos[2]
                ];
                state.camera.quat = Quat.fromLookRotation(fwd, [0, 0, 1]);
            }
            btnFly.classList.add('active');
            btnOrbit.classList.remove('active');
            flyHelp.classList.remove('hidden');
            orbitHelp.classList.add('hidden');
        });

        btnOrbit.addEventListener('click', () => {
            state.cameraMode = 'orbit';
            const pos = state.camera.pos;
            state.camera.orbitDistance = Math.hypot(pos[0], pos[1], pos[2]);
            state.camera.orbitTheta = Math.atan2(pos[1], pos[0]);
            state.camera.orbitPhi = Math.asin(Math.max(-1.0, Math.min(1.0, pos[2] / state.camera.orbitDistance)));
            btnOrbit.classList.add('active');
            btnFly.classList.remove('active');
            orbitHelp.classList.remove('hidden');
            flyHelp.classList.add('hidden');
        });

        const btnFrame = document.getElementById('btn-frame-planet');
        if (btnFrame) {
            btnFrame.addEventListener('click', () => {
                framePlanet();
            });
        }

        const sliderScale = document.getElementById('slider-scale');
        if (sliderScale) {
            sliderScale.addEventListener('input', (e) => {
                state.scaleMultiplier = parseFloat(e.target.value);
                document.getElementById('val-scale').innerText = state.scaleMultiplier.toFixed(2) + 'x';
            });
        }

        const sliderThickness = document.getElementById('slider-thickness');
        if (sliderThickness) {
            sliderThickness.addEventListener('input', (e) => {
                state.thicknessMultiplier = parseFloat(e.target.value);
                document.getElementById('val-thickness').innerText = state.thicknessMultiplier.toFixed(2) + 'x';
            });
        }

        const sliderAlpha = document.getElementById('slider-alpha');
        sliderAlpha.addEventListener('input', (e) => {
            state.alphaBoost = parseFloat(e.target.value);
            document.getElementById('val-alpha').innerText = state.alphaBoost.toFixed(1) + 'x';
        });

        const sliderSpeed = document.getElementById('slider-speed');
        sliderSpeed.addEventListener('input', (e) => {
            state.flightSpeed = parseFloat(e.target.value);
            document.getElementById('val-speed').innerText = state.flightSpeed.toFixed(0) + ' km/s';
        });

        const sliderLodBias = document.getElementById('slider-lod-bias');
        if (sliderLodBias) {
            sliderLodBias.addEventListener('input', (e) => {
                state.lodBias = parseFloat(e.target.value);
                document.getElementById('val-lod-bias').innerText = state.lodBias.toFixed(1) + 'x';
            });
        }

        const selectShading = document.getElementById('select-shading');
        if (selectShading) {
            selectShading.addEventListener('change', (e) => {
                state.colorMode = parseInt(e.target.value, 10);
            });
        }

        const selectKernel = document.getElementById('select-kernel');
        if (selectKernel) {
            selectKernel.addEventListener('change', (e) => {
                state.kernelType = parseInt(e.target.value, 10);
            });
        }

        const checkSunShading = document.getElementById('check-sun-shading');
        if (checkSunShading) {
            checkSunShading.addEventListener('change', (e) => {
                state.shadingEnabled = e.target.checked;
            });
        }

        const sliderSunAzimuth = document.getElementById('slider-sun-azimuth');
        if (sliderSunAzimuth) {
            sliderSunAzimuth.addEventListener('input', (e) => {
                state.sunAzimuth = parseFloat(e.target.value);
                document.getElementById('val-sun-azimuth').innerText = state.sunAzimuth.toFixed(0) + '°';
            });
        }

        const sliderSunElevation = document.getElementById('slider-sun-elevation');
        if (sliderSunElevation) {
            sliderSunElevation.addEventListener('input', (e) => {
                state.sunElevation = parseFloat(e.target.value);
                document.getElementById('val-sun-elevation').innerText = state.sunElevation.toFixed(0) + '°';
            });
        }

        const sliderHazeDensity = document.getElementById('slider-haze-density');
        if (sliderHazeDensity) {
            sliderHazeDensity.addEventListener('input', (e) => {
                state.hazeDensity = parseFloat(e.target.value);
                document.getElementById('val-haze-density').innerText = state.hazeDensity.toFixed(2) + 'x';
            });
        }

        const selectPreset = document.getElementById('select-terrain-preset');
        if (selectPreset) {
            selectPreset.addEventListener('change', (e) => {
                const preset = e.target.value;
                const R = parseFloat(document.getElementById('gen-radius').value) || 100.0;
                if (preset === 'earth') {
                    document.getElementById('gen-elev').value = (R * 0.08).toFixed(1);
                    document.getElementById('gen-elev-offset').value = '0.00';
                    document.getElementById('gen-freq').value = '1.5';
                } else if (preset === 'highlands') {
                    document.getElementById('gen-elev').value = (R * 0.18).toFixed(1);
                    document.getElementById('gen-elev-offset').value = '0.15';
                    document.getElementById('gen-freq').value = '2.2';
                } else if (preset === 'archipelago') {
                    document.getElementById('gen-elev').value = (R * 0.06).toFixed(1);
                    document.getElementById('gen-elev-offset').value = '-0.25';
                    document.getElementById('gen-freq').value = '3.5';
                } else if (preset === 'pangaea') {
                    document.getElementById('gen-elev').value = (R * 0.12).toFixed(1);
                    document.getElementById('gen-elev-offset').value = '0.10';
                    document.getElementById('gen-freq').value = '0.8';
                } else if (preset === 'smooth') {
                    document.getElementById('gen-elev').value = (R * 0.01).toFixed(1);
                    document.getElementById('gen-elev-offset').value = '0.00';
                    document.getElementById('gen-freq').value = '1.0';
                }
            });
        }

        document.getElementById('check-spheroid-grid').addEventListener('change', (e) => {
            state.showGrid = e.target.checked;
        });

        const genOverlap = document.getElementById('gen-overlap');
        genOverlap.addEventListener('input', (e) => {
            document.getElementById('val-gen-overlap').innerText = parseFloat(e.target.value).toFixed(2);
        });

        const sliderCurvTol = document.getElementById('slider-curv-tol');
        if (sliderCurvTol) {
            sliderCurvTol.addEventListener('input', (e) => {
                document.getElementById('val-curv-tol').innerText = parseFloat(e.target.value).toFixed(2) + ' km';
            });
        }

        const btnRegenerate = document.getElementById('btn-regenerate');
        btnRegenerate.addEventListener('click', async () => {
            btnRegenerate.disabled = true;
            showLoader('Generating Chunked Multi-LOD Gaussian Splats...');

            const gridStr = document.getElementById('gen-chunk-grid') ? document.getElementById('gen-chunk-grid').value : 'auto';
            let geometryType = 'cubed-sphere';
            let patchesPerFace = 0; // 0 triggers auto-adaptive density chunking
            let chunkRows = 8;
            let chunkCols = 16;

            if (gridStr === 'auto') {
                geometryType = 'cubed-sphere';
                patchesPerFace = 0;
            } else if (gridStr.startsWith('cubed-')) {
                geometryType = 'cubed-sphere';
                patchesPerFace = parseInt(gridStr.replace('cubed-', ''), 10) || 4;
            } else {
                geometryType = 'latlon';
                const parts = gridStr.split('x');
                chunkRows = parseInt(parts[0], 10) || 8;
                chunkCols = parseInt(parts[1], 10) || 16;
            }

            const payload = {
                geometryType: geometryType,
                patchesPerFace: patchesPerFace,
                radius: parseFloat(document.getElementById('gen-radius').value),
                flattening: parseFloat(document.getElementById('gen-flattening').value),
                latMin: -90.0,
                latMax: 90.0,
                lonMin: -180.0,
                lonMax: 180.0,
                stepsLat: parseInt(document.getElementById('gen-res-lat').value, 10),
                stepsLon: parseInt(document.getElementById('gen-res-lon').value, 10),
                elevationScale: parseFloat(document.getElementById('gen-elev').value),
                elevationOffset: parseFloat(document.getElementById('gen-elev-offset').value),
                noiseSeed: parseInt(document.getElementById('gen-seed').value, 10),
                noiseFrequency: parseFloat(document.getElementById('gen-freq').value),
                octaves: 5,
                overlapFactor: parseFloat(document.getElementById('gen-overlap').value),
                thicknessFactor: 0.15,
                seaLevel: 0.30,
                adaptiveDensity: document.getElementById('check-adaptive-density') ? document.getElementById('check-adaptive-density').checked : true,
                curvatureTolerance: document.getElementById('slider-curv-tol') ? parseFloat(document.getElementById('slider-curv-tol').value) : 0.12,
                chunkRows: chunkRows,
                chunkCols: chunkCols
            };

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    await fetchSplatData();
                }
            } catch (err) {
                alert('Regeneration failed: ' + err.message);
            } finally {
                btnRegenerate.disabled = false;
                hideLoader();
            }
        });
    }

    function showLoader(msg) {
        const loader = document.getElementById('loader');
        document.getElementById('loader-msg').innerText = msg;
        loader.classList.remove('hidden');
    }

    function hideLoader() {
        const loader = document.getElementById('loader');
        loader.classList.add('hidden');
    }

    initGL();
    setupUI();
    fetchSplatData();
    requestAnimationFrame(render);

})();
