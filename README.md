# 3D Gaussian Splat Oblate Spheroid Terrain Spike

A high-performance Go spike investigating **geometry-less 3D Gaussian Splatting** for planetary & GIS terrain on **oblate spheroids** (non-flat ellipsoids of revolution such as WGS-84).

---

## 🌟 Key Capabilities

1. **Oblate Spheroid Geodesy & Math (`pkg/spheroid`)**:
   - WGS-84 reference ellipsoid and scalable model spheroids with arbitrary flattening $f = (a - b) / a$.
   - Geodetic $(\text{lat}, \text{lon}, h) \leftrightarrow \text{ECEF } (X, Y, Z)$ coordinate transformations.
   - Surface normals and orthonormal East-North-Up (ENU) tangent frames.
   - Normal-aligned rotation quaternions that orient each Gaussian ellipse tangentially along the curved spheroid surface with local terrain slope adjustments.

2. **Procedural GIS Stand-In Generator (`pkg/terrain`, `cmd/generate`)**:
   - Generates 3D Gaussian Splats over arbitrary geodetic bounds.
   - Multi-octave fractal elevation noise (continental bases, mountain ridges, shorelines).
   - Anisotropic splat sizing matching local metric grid spacing ($\Delta s_{\text{east}}, \Delta s_{\text{north}}$) with polar meridian convergence compensation and configurable overlap factors for seamless geometry-less coverage.
   - Biome color synthesis (ocean, coast, beach, grass, highland forest, mountain rock, snow peaks).

3. **Standard 3DGS Serialization (`pkg/splat`)**:
   - Standard 32-byte binary `.splat` format (compatible with web and GPU pipelines).
   - Standard 3D Gaussian Splatting binary `.ply` format with spherical harmonics DC and log scales (compatible with external 3DGS tools).

4. **Interactive 3D Splat Fly-Cam Viewer (`pkg/viewer`, `cmd/viewer`)**:
   - WebGL 2.0 3D Gaussian Splat renderer running at 60+ FPS.
   - **Fly Camera (6-DOF)**: WASD directional movement, Space/Shift altitude control, mouse drag look, and speed controls.
   - **Orbit Camera**: Inspecting the planetary spheroid from space.
   - Real-time geodetic telemetry HUD (Latitude, Longitude, Altitude above spheroid).
   - Live parameter tuning: Splat scale multiplier, alpha boost, colormaps (Biomes, Elevation, Normals, Footprints), and live terrain regeneration.

---

## 🚀 Quick Start

### 1. Generate Terrain Splats
```bash
# Generate 45,000 splats on an oblate spheroid (saved to terrain.splat)
go run ./cmd/generate -out terrain.splat -res-lat 150 -res-lon 300 -radius 100 -flattening 0.05

# Or generate a standard 3DGS binary PLY file
go run ./cmd/generate -out terrain.ply -format ply
```

#### Generator Flags:
| Flag | Default | Description |
|---|---|---|
| `-out` | `terrain.splat` | Output path (`.splat` or `.ply`) |
| `-radius` | `100.0` | Equatorial radius $a$ |
| `-flattening` | `0.05` | Flattening $f = (a - b) / a$ (0 for sphere) |
| `-lat-min`, `-lat-max` | `-90.0`, `90.0` | Latitude extent in degrees (Poles to Poles) |
| `-lon-min`, `-lon-max` | `-180.0`, `180.0` | Longitude extent in degrees (Full 360° Globe) |
| `-res-lat`, `-res-lon` | `150`, `300` | Resolution (Lat rings & Equator base for Equal-Area sampling) |
| `-elev-scale` | `8.0` | Peak elevation height (km/units) |
| `-elev-offset` | `0.0` | Baseline elevation shift (-0.5 to +0.5 to change land/sea ratio) |
| `-overlap` | `1.35` | Splat tangent overlap multiplier |
| `-thickness` | `0.15` | Normal thickness ratio (surface-flat alignment) |

---

### 2. Launch Interactive 3D Viewer
```bash
# Start viewer and automatically open browser at http://localhost:8080
go run ./cmd/viewer -file terrain.splat -port 8080
```

#### Viewer Navigation Controls:
- **Fly Mode**:
  - `W` / `S`: Move Forward / Backward
  - `A` / `D`: Strafe Left / Right
  - `Space` / `Shift`: Climb Up / Dive Down
  - `Mouse Drag`: Pitch & Yaw free look
  - `Mouse Wheel` / `Slider`: Adjust flight velocity
- **Orbit Mode**:
  - `Left Click + Drag`: Orbit around spheroid
  - `Mouse Wheel`: Zoom In / Out
- **HUD & Tuning Sidebar**:
  - Tweak Splat Scale & Alpha in real time to test continuous coverage vs footprint separation
  - Switch Shading: Biomes $\to$ Elevation Map $\to$ Normal Vectors $\to$ Gaussian Contours
  - Live Regenerator: Change oblateness, density, or noise seed and update on the fly

---

## 🧪 Testing

Run all unit tests across all packages:
```bash
go test ./... -v
```
