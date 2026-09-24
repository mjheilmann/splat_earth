# 3D Gaussian Splat Oblate Spheroid Terrain Spike

A high-performance Go spike investigating **geometry-less 3D Gaussian Splatting** for planetary & GIS terrain on **oblate spheroids** (non-flat ellipsoids of revolution such as WGS-84).

---

## 🌟 Key Capabilities

1. **Oblate Spheroid Geodesy & Math (`pkg/spheroid`)**:
   - WGS-84 reference ellipsoid and scalable model spheroids with arbitrary flattening $f = (a - b) / a$.
   - Geodetic $(\text{lat}, \text{lon}, h) \leftrightarrow \text{ECEF } (X, Y, Z)$ coordinate transformations.
   - Surface normals and orthonormal East-North-Up (ENU) tangent frames.
   - Normal-aligned rotation quaternions that orient each Gaussian ellipse tangentially along the curved spheroid surface with local terrain slope adjustments.

2. **Data-Driven DEM Topography Engine (`pkg/terrain`, `cmd/generate`)**:
   - **Real-World Earth DEM**: Directly samples high-resolution global elevation and ocean bathymetry from equirectangular rasters such as the [Wikimedia Commons World Elevation Map](https://commons.wikimedia.org/wiki/File:World_elevation_map.png) (21,600 × 10,800 1-arcminute global grid).
   - **Steered Multi-Resolution Focus Areas**: Concentrates ultra-dense micro-splats in Areas of Interest (e.g. 10 sq miles, 64x–128x density, ~50m cell spacing) while keeping the global planet at base density (~1.8 km spacing) with smooth trigonometric boundary transitions and zero edge seams.
   - Anisotropic splat sizing matching local metric grid spacing with polar meridian convergence compensation.
   - Satellite-like biome color synthesis (deep ocean trenches, continental shelves, beaches, plains, highlands, craggy rock, and glacial snowcaps).

3. **Standard 3DGS Serialization (`pkg/splat`)**:
   - Standard 32-byte binary `.splat` format (compatible with web and GPU pipelines).
   - Standard 3D Gaussian Splatting binary `.ply` format with spherical harmonics DC and log scales (compatible with external 3DGS tools).

4. **Interactive 3D Splat Fly-Cam Viewer (`pkg/viewer`, `cmd/viewer`)**:
   - WebGL 2.0 3D Gaussian Splat renderer running at 60+ FPS with chunked multi-LOD spatial streaming.
   - **Fly Camera (6-DOF)**: WASD directional movement, Space/Shift altitude control, mouse drag look, and speed controls.
   - **Orbit Camera**: Inspecting the planetary spheroid from orbit.
   - Real-time geodetic telemetry HUD (Latitude, Longitude, Altitude above spheroid, Global vs Focus Spacing).
   - Geographic landmark navigation shortcuts (Himalayas/Everest, Mariana Trench, Grand Canyon, Alps, Hawaii, Amazon Basin).

---

### Global Elevation & Bathymetry Base Map
The engine supports direct sampling from the high-resolution global elevation and ocean bathymetry raster available on Wikimedia Commons:

- **Source**: [Wikimedia Commons - File:World elevation map.png](https://commons.wikimedia.org/wiki/File:World_elevation_map.png)
- **Resolution**: 21,600 × 10,800 pixels (~1.85 km per pixel at the equator)
- **Format**: Equirectangular projection (covering −180° to +180° Longitude and −90° to +90° Latitude).
- **Values**: Grayscale elevation where `0` corresponds to Mariana Trench (~−11,000 m), `~143` corresponds to Sea Level (0 m), and `255` corresponds to Mt. Everest (+8,848 m).

### High-Resolution Focus Areas (USGS 3DEP LiDAR DEMs)
For ultra-high resolution focus zones (down to 1-meter physical resolution), download 1-meter GeoTIFF DEM tiles from:

- **Source**: [USGS 3DEP LiDAR Explorer](https://apps.nationalmap.gov/lidar-explorer/#/)
- **Resolution**: 1.0 meter / pixel (floating-point GeoTIFF elevation rasters).
- **Usage**: Drop downloaded `.tif` DEM files into the `images/` folder. The generator automatically detects any DEM tiles present, extracts their geographic bounding boxes (via UTM Zone $\leftrightarrow$ Geodetic projection), and blends them as high-density LiDAR focus layers over the global world heightmap.

> **Note**: `images/` (including `World_elevation_map.png` and all `.tif` files) and `splats/` are ignored in `.gitignore`.

---

## 🚀 Quick Start

### 1. Generate Terrain Splats
```bash
# Generate real Earth elevation splats from the DEM map (saved to terrain.splat)
go run ./cmd/generate -raster images/World_elevation_map.png -out terrain.splat -res-lat 150 -res-lon 300

# Or generate with a steered 10 sq mile high-resolution focus area in the Himalayas:
go run ./cmd/generate -out terrain.splat -focus-enable -focus-lat 27.98 -focus-lon 86.92 -focus-area 10.0 -focus-boost 64

# Or generate a standard 3DGS binary PLY file
go run ./cmd/generate -out terrain.ply -format ply
```

#### Generator Flags:
| Flag | Default | Description |
|---|---|---|
| `-out` | `terrain.splat` | Output path (`.splat` or `.ply`) |
| `-raster` | `images/World_elevation_map.png` | Path to equirectangular DEM raster |
| `-radius` | `100.0` | Equatorial radius $a$ (km) |
| `-flattening` | `0.05` | Flattening $f = (a - b) / a$ (0 for sphere, 0.00335 for WGS-84) |
| `-lat-min`, `-lat-max` | `-90.0`, `90.0` | Latitude extent in degrees (Poles to Poles) |
| `-lon-min`, `-lon-max` | `-180.0`, `180.0` | Longitude extent in degrees (Full 360° Globe) |
| `-res-lat`, `-res-lon` | `150`, `300` | Resolution (Lat rings & Equator base for Equal-Area sampling) |
| `-elev-scale` | `12.0` | Peak elevation height (km) |
| `-overlap` | `1.35` | Splat tangent overlap multiplier |
| `-thickness` | `0.035` | Normal thickness ratio (surface-flat disc alignment) |
| `-focus-enable` | `false` | Enable steered high-density focus area (default 10 sq miles) |
| `-focus-lat`, `-focus-lon` | `0.0`, `0.0` | Focus area center coordinates |
| `-focus-area` | `10.0` | Focus area size in square miles |
| `-focus-boost` | `64.0` | Density boost multiplier in focus area (e.g. 16x - 128x) |

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
  - `Q` / `E`: Roll Left / Right (6-DOF, no gimbal lock)
  - `Space` / `C`: Climb Up / Dive Down
  - `Mouse Drag`: Pitch & Yaw free look
  - `Mouse Wheel` / `Slider`: Adjust flight velocity
- **Orbit Mode**:
  - `Left Click + Drag`: Orbit around spheroid
  - `Mouse Wheel`: Zoom In / Out
- **HUD & Tuning Sidebar**:
  - **Color Mode**: Realistic Biomes, Elevation Hypsometric Colormap, Surface Normals, Gaussian Footprint Debug
  - **Lighting**: Dynamic Sun Shading, Azimuth, Altitude, Atmospheric Haze
  - **Navigation Shortcuts**: Quick jump to Everest, Mariana Trench, Grand Canyon, Alps, Hawaii, Amazon Basin
