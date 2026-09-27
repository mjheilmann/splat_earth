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

### Supported Elevation & Imagery Datasets

The engine features an automatic GeoData Catalog (`pkg/terrain/catalog.go`) that scans the `images/` folder, parses GeoTIFF geokeys, bounding boxes, and projections, and blends the highest-resolution color and topography layers available.

#### 1. NASA Blue Marble: Next Generation (500m Global True Color & Topo/Bathy)
Directly consume NASA's high-resolution global 500m GeoTIFF imagery and topography tiles:
- **Source**: [NASA Earth Observatory - Blue Marble: Next Generation](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/)
- **Base Topography & Bathymetry**: [NASA Blue Marble Base Topography/Bathymetry](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography-bathymetry/)
- **Resolution**: 500 meters / pixel ($21,600 \times 21,600$ per quadrant, $86,400 \times 43,200$ global).
- **Format**: 8-quadrant GeoTIFF tiles (A1, A2, B1, B2, C1, C2, D1, D2).
- **Usage**: Drop any or all 8 `world.topo.bathy.*_geo.tif` tiles into the `images/` directory. The engine auto-discovers them and maps 24-bit True Color satellite photorealism directly onto the splats.

#### 2. GEBCO Global Ocean Bathymetry & Land Elevation (1km Grid)
- **Source**: [GEBCO (General Bathymetric Chart of the Oceans)](https://www.gebco.net/)
- **Resolution**: ~1 km grid ($43,200 \times 21,600$ global).
- **Format**: 8-quadrant elevation (`gebco_08_rev_elev_*_grey_geo.tif`) and bathymetry (`gebco_08_rev_bath_*_grey_geo.tif`) GeoTIFFs.
- **Usage**: Drop into `images/` for sub-pixel bilinear-interpolated terrestrial elevation and ocean trench topography.

#### 3. Global Elevation & Bathymetry Fallback Map
- **Source**: [Wikimedia Commons - File:World elevation map.png](https://commons.wikimedia.org/wiki/File:World_elevation_map.png)
- **Resolution**: 21,600 × 10,800 pixels (~1.85 km per pixel at the equator).
- **Format**: Equirectangular global grayscale elevation map (0 = −11,000m Mariana Trench, 143 = Sea Level, 255 = +8,848m Mt. Everest).

#### 4. High-Resolution Focus Areas (USGS 3DEP 1-Meter LiDAR DEMs)
For ultra-high resolution focus zones (down to 1-meter physical resolution):
- **Source**: [USGS 3DEP LiDAR Explorer](https://apps.nationalmap.gov/lidar-explorer/#/)
- **Resolution**: 1.0 meter / pixel (floating-point GeoTIFF elevation rasters).
- **Usage**: Drop downloaded `.tif` DEM files into the `images/` folder. The generator automatically detects any DEM tiles present, extracts their geographic bounding boxes (via UTM Zone $\leftrightarrow$ Geodetic projection), and blends them as high-density LiDAR focus layers over the global world heightmap.

> **Note**: The `images/` folder and `splats/` output directory are excluded in `.gitignore` so large multi-gigabyte datasets are never committed to git.

---

## 🚀 Quick Start

### 1. Build and Generate Terrain Splats
```bash
# Build the generator and viewer binaries
go build ./cmd/generate && go build ./cmd/viewer

# Option A: Standard Dramatic Relief (3.2x vertical exaggeration, 1.0 km height scale)
./generate -out splats/world.splat

# Option B: True 1:1 Physical Scale (Authentic unexaggerated real-world Earth geometry)
./generate -elev-scale 0.311 -out splats/world.splat

# Option C: Fast low-resolution preview
./generate -res-lat 150 -res-lon 300 -out splats/preview.splat
```

#### Generator Flags:
| Flag | Default | Description |
|---|---|---|
| `-out` | `splats/terrain.splat` | Output path (`.splat` multi-LOD container or `.ply`) |
| `-elev-scale` | `1.0` | Peak elevation scale in km (`1.0` = 3.2x dramatic relief, `0.311` = true 1:1 physical scale) |
| `-elev-offset` | `0.0` | Baseline elevation offset (-0.5 to +0.5) |
| `-radius` | `100.0` | Equatorial radius $a$ in km (WGS-84 oblate spheroid model) |
| `-flattening` | `0.0033528` | WGS-84 standard flattening ratio $f = (a - b) / a$ (0.335%) |
| `-res-lat`, `-res-lon` | `0` (Auto) | Global raster grid resolution (`0` = Auto native source limit, e.g. $21,600 \times 43,200$) |
| `-patches` | `0` (Auto) | Patches per face axis (`0` = Auto optimal, e.g. `8` => 384 chunks) |
| `-overlap` | `1.38` | Splat footprint overlap multiplier (gapless coverage) |
| `-thickness` | `0.035` | Normal thickness ratio (surface-flat disc alignment) |
| `-raster` | `images/World_elevation_map.png` | Fallback DEM raster path |
| `-biome-config` | `config/biomes.json` | Path to JSON biome hypsometry & color tuning config |

---

### 2. Launch Interactive 3D Viewer
```bash
# Start viewer (auto-detects splats/terrain.splat or splats/world.splat)
./viewer -port 8080
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
