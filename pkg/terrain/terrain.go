package terrain

import (
	"math"
	"runtime"
	"sync"

	"github.com/mjheilmann/gaussian-splat-app/pkg/spheroid"
	"github.com/mjheilmann/gaussian-splat-app/pkg/splat"
)

// DensityZone defines a steered high-density area of interest (AOI) on the spheroid.
type DensityZone struct {
	Enabled           bool    `json:"enabled"`
	CenterLat         float64 `json:"centerLat"`         // Latitude in degrees (-90 to 90)
	CenterLon         float64 `json:"centerLon"`         // Longitude in degrees (-180 to 180)
	AreaSqMiles       float64 `json:"areaSqMiles"`       // Area in square miles (default: 10.0 sq miles)
	DensityMultiplier float64 `json:"densityMultiplier"` // Density boost factor (e.g. 8.0x - 64.0x)
	TransitionWidthKm float64 `json:"transitionWidthKm"` // Smooth boundary transition margin in km (0 = auto)
}

// RadiusKm returns the equivalent circular radius in km for the configured area in square miles.
func (dz DensityZone) RadiusKm() float64 {
	areaSqMiles := dz.AreaSqMiles
	if areaSqMiles <= 0 {
		areaSqMiles = 10.0
	}
	areaSqKm := areaSqMiles * spheroid.SqMilesToSqKm
	return math.Sqrt(areaSqKm / math.Pi)
}

// GeneratorConfig holds configuration for the data-driven terrain generator.
type GeneratorConfig struct {
	Spheroid spheroid.Spheroid // Reference ellipsoid

	LatMin float64 // Minimum latitude in degrees (-90 to 90)
	LatMax float64 // Maximum latitude in degrees (-90 to 90)
	LonMin float64 // Minimum longitude in degrees (-180 to 180)
	LonMax float64 // Maximum longitude in degrees (-180 to 180)

	StepsLat int // Number of latitude rings from pole to pole
	StepsLon int // Equator reference resolution for longitude

	ElevationScale  float64 // Max elevation height above ellipsoid (km or units)
	ElevationOffset float64 // Baseline elevation offset (-0.5 to 0.5)
	BaseElevation   float64 // Sea level / baseline offset

	OverlapFactor      float64 // Splat size overlap multiplier (e.g. 1.25 - 1.4)
	ThicknessFactor    float64 // Ratio of vertical normal scale to horizontal tangent scale (e.g. 0.035)
	SeaLevel           float64 // Relative height threshold for water (0.0 - 1.0)
	AdaptiveDensity    bool    // Enable quadtree curvature-adaptive variable density
	CurvatureTolerance float64 // Height deviation threshold in km/units to subdivide quad cells (e.g. 0.12 km)

	RasterPath   string   `json:"rasterPath,omitempty"`   // Path to real-world DEM raster image (e.g. "images/World_elevation_map.png")
	GEBCOPaths   []string `json:"gebcoPaths,omitempty"`   // Paths to 1km global GEBCO GeoTIFF quadrant tiles (e.g. "images/gebco_08_*.tif")
	GeoTIFFPaths []string `json:"geotiffPaths,omitempty"` // Paths to high-resolution local GeoTIFF LiDAR DEMs (e.g. "images/USGS_1M_*.tif")

	BiomeConfigPath string      `json:"biomeConfigPath,omitempty"` // Path to custom JSON biome configuration file (e.g. "config/biomes.json")
	Biome           BiomeConfig `json:"biome"`                     // Loaded biome hypsometric configuration

	FocusArea  DensityZone   `json:"focusArea"`  // Primary steered high-density area of interest
	FocusAreas []DensityZone `json:"focusAreas"` // Optional multiple steered focus areas
}

// ActiveFocusAreas returns a slice of all currently enabled focus zones.
func (cfg *GeneratorConfig) ActiveFocusAreas() []DensityZone {
	var active []DensityZone
	if cfg.FocusArea.Enabled {
		active = append(active, cfg.FocusArea)
	}
	for _, fa := range cfg.FocusAreas {
		if fa.Enabled {
			active = append(active, fa)
		}
	}
	return active
}

// DefaultConfig returns parameters for a complete oblate spheroid terrain.
func DefaultConfig() GeneratorConfig {
	rasterPath := FindDefaultElevationRaster()
	gebcoFiles := FindDefaultGEBCOFiles()
	geoTIFFs := FindDefaultGeoTIFFs()
	biomePath := FindDefaultBiomeConfig()

	biomeCfg := DefaultBiomeConfig()
	if biomePath != "" {
		if loaded, err := LoadBiomeConfigFile(biomePath); err == nil {
			biomeCfg = loaded
		}
	}

	seaLevel := 0.30
	if rasterPath != "" || len(gebcoFiles) > 0 || len(geoTIFFs) > 0 {
		if biomeCfg.SeaLevel > 0 {
			seaLevel = biomeCfg.SeaLevel
		} else {
			seaLevel = 143.35 / 255.0 // ~0.56215 true sea level in standard DEM
		}
	}

	focusArea := DensityZone{
		Enabled:           false,
		CenterLat:         0.0,
		CenterLon:         0.0,
		AreaSqMiles:       10.0,
		DensityMultiplier: 64.0,
		TransitionWidthKm: 0.0,
	}

	// Auto-configure focus area to match GeoTIFF bounds if LiDAR files are detected
	if len(geoTIFFs) > 0 {
		focusArea.CenterLat = 47.581
		focusArea.CenterLon = -122.335
		focusArea.AreaSqMiles = 154.0 // ~400 km2 Seattle mosaic
		focusArea.DensityMultiplier = 64.0
	}

	stepsLat := 150
	stepsLon := 300
	if len(gebcoFiles) >= 8 {
		stepsLat = 21600
		stepsLon = 43200
	}

	return GeneratorConfig{
		Spheroid:           spheroid.NewSpheroid(100.0, spheroid.WGS84Flattening), // Scaled spheroid: radius 100, WGS-84 flattening (0.335%)
		LatMin:             -90.0,
		LatMax:             90.0,
		LonMin:             -180.0,
		LonMax:             180.0,
		StepsLat:           stepsLat,
		StepsLon:           stepsLon,
		ElevationScale:     1.0, // Realistic elevation scale: 1.0 km visual relief (0.311 km for true 1:1 physical scale)
		ElevationOffset:    0.0,
		BaseElevation:      0.0,
		OverlapFactor:      1.35,
		ThicknessFactor:    0.035, // Wafer-thin planar discs (~3.5% of cell width)
		SeaLevel:           seaLevel,
		AdaptiveDensity:    false,
		CurvatureTolerance: 0.12,
		RasterPath:         rasterPath,
		GEBCOPaths:         gebcoFiles,
		GeoTIFFPaths:       geoTIFFs,
		BiomeConfigPath:    biomePath,
		Biome:              biomeCfg,
		FocusArea:          focusArea,
		FocusAreas:         nil,
	}
}

// TerrainGenerator generates 3D Gaussian Splats over an oblate spheroid.
type TerrainGenerator struct {
	config  GeneratorConfig
	raster  *ElevationRaster
	gebco   *GEBCOMosaic
	geoTIFF *GeoTIFFMosaic
}

// GetDensityMultiplier returns the steered density boost factor at (lat, lon).
func (tg *TerrainGenerator) GetDensityMultiplier(lat, lon float64) float64 {
	activeZones := tg.config.ActiveFocusAreas()
	maxMult := 1.0

	// 1. Direct GeoTIFF DEM file bounding box matching
	if tg.geoTIFF != nil && tg.geoTIFF.Contains(lat, lon) {
		maxMult = 64.0
		if len(activeZones) > 0 && activeZones[0].DensityMultiplier > maxMult {
			maxMult = activeZones[0].DensityMultiplier
		}
	}

	// 2. Custom active focus zones
	for _, zone := range activeZones {
		m := zone.DensityMultiplier
		if m < 1.0 {
			m = 1.0
		}
		radius := zone.RadiusKm()
		trans := zone.TransitionWidthKm
		if trans <= 0 {
			trans = math.Max(0.5, radius*0.5)
		}

		dist := tg.config.Spheroid.SurfaceDistance(lat, lon, zone.CenterLat, zone.CenterLon)
		if dist <= radius {
			if m > maxMult {
				maxMult = m
			}
		} else if dist < radius+trans {
			t := (dist - radius) / trans
			// Cosine smooth falloff
			w := 0.5 * (1.0 + math.Cos(math.Pi*t))
			val := 1.0 + (m-1.0)*w
			if val > maxMult {
				maxMult = val
			}
		}
	}
	return maxMult
}

// NewTerrainGenerator initializes a new data-driven terrain generator.
func NewTerrainGenerator(cfg GeneratorConfig) *TerrainGenerator {
	if cfg.OverlapFactor <= 0 {
		cfg.OverlapFactor = 1.35
	}
	if cfg.ThicknessFactor <= 0 {
		cfg.ThicknessFactor = 0.035
	}
	if cfg.StepsLat < 2 {
		cfg.StepsLat = 2
	}
	if cfg.StepsLon < 2 {
		cfg.StepsLon = 2
	}

	var gebco *GEBCOMosaic
	gebcoPaths := cfg.GEBCOPaths
	if len(gebcoPaths) == 0 {
		gebcoPaths = FindDefaultGEBCOFiles()
	}
	if len(gebcoPaths) >= 8 {
		if m, err := LoadGEBCOMosaic(gebcoPaths); err == nil {
			gebco = m
		}
	}

	var raster *ElevationRaster
	rasterPath := cfg.RasterPath
	if rasterPath == "" && gebco == nil {
		rasterPath = FindDefaultElevationRaster()
	}

	if rasterPath != "" {
		if r, err := LoadElevationRaster(rasterPath); err == nil {
			raster = r
			if cfg.SeaLevel == 0.30 || cfg.SeaLevel == 0.0 {
				cfg.SeaLevel = 143.0 / 255.0 // ~0.5607 true sea level in standard DEM
			}
		}
	}

	var geoTIFF *GeoTIFFMosaic
	geoTIFFPaths := cfg.GeoTIFFPaths
	if len(geoTIFFPaths) == 0 {
		geoTIFFPaths = FindDefaultGeoTIFFs()
	}
	if len(geoTIFFPaths) > 0 {
		if m, err := LoadGeoTIFFMosaic(geoTIFFPaths); err == nil {
			geoTIFF = m
		}
	}

	if cfg.BiomeConfigPath != "" {
		if loaded, err := LoadBiomeConfigFile(cfg.BiomeConfigPath); err == nil {
			cfg.Biome = loaded
		}
	} else if len(cfg.Biome.LandBands) == 0 {
		biomePath := FindDefaultBiomeConfig()
		if biomePath != "" {
			if loaded, err := LoadBiomeConfigFile(biomePath); err == nil {
				cfg.Biome = loaded
			} else {
				cfg.Biome = DefaultBiomeConfig()
			}
		} else {
			cfg.Biome = DefaultBiomeConfig()
		}
	}

	if cfg.Biome.SeaLevel > 0 {
		cfg.SeaLevel = cfg.Biome.SeaLevel
	}

	return &TerrainGenerator{
		config:  cfg,
		raster:  raster,
		gebco:   gebco,
		geoTIFF: geoTIFF,
	}
}

// Generate creates a slice of 3D Gaussian Splats using Equal-Area adaptive sampling.
// Splat generation is parallelized across all CPU cores.
func (tg *TerrainGenerator) Generate() []splat.Splat {
	cfg := tg.config
	dLat := (cfg.LatMax - cfg.LatMin) / float64(cfg.StepsLat-1)
	dLatRad := dLat * math.Pi / 180.0
	targetSpacing := (2.0 * math.Pi * cfg.Spheroid.SemiMajorAxis) / float64(cfg.StepsLon)

	ringSplats := make([][]splat.Splat, cfg.StepsLat)

	type ringTask struct {
		ringIdx int
		lat     float64
		latRad  float64
	}

	tasks := make(chan ringTask, cfg.StepsLat)
	for i := 0; i < cfg.StepsLat; i++ {
		lat := cfg.LatMin + float64(i)*dLat
		latRad := lat * math.Pi / 180.0
		tasks <- ringTask{ringIdx: i, lat: lat, latRad: latRad}
	}
	close(tasks)

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers < 1 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tasks {
				i := task.ringIdx
				lat := task.lat
				latRad := task.latRad

				if math.Abs(lat) >= 89.99 {
					elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(lat, 0)
					height := cfg.BaseElevation + (elevNorm-cfg.SeaLevel+cfg.ElevationOffset)*cfg.ElevationScale
					posECEF := cfg.Spheroid.GeodeticToECEF(lat, 0, height)

					capRadius := float32(0.5 * targetSpacing * cfg.OverlapFactor * 1.2)
					quat := cfg.Spheroid.OrientationQuaternion(lat, 0, slopeE*0.1, slopeN*0.1)
					color := tg.computeBiomeColor(lat, 0, elevNorm, slopeE, slopeN)

					ringSplats[i] = []splat.Splat{{
						Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
						Scale:    [3]float32{capRadius, capRadius, capRadius * float32(cfg.ThicknessFactor)},
						Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
						Color:    color,
					}}
					continue
				}

				sinLat := math.Sin(latRad)
				cosLat := math.Cos(latRad)

				denom := math.Sqrt(1.0 - cfg.Spheroid.Eccentricity2*sinLat*sinLat)
				N := cfg.Spheroid.SemiMajorAxis / denom
				M := cfg.Spheroid.SemiMajorAxis * (1.0 - cfg.Spheroid.Eccentricity2) / (denom * denom * denom)

				dNorthMeters := M * dLatRad
				parallelCircumference := 2.0 * math.Pi * N * math.Abs(cosLat)

				nLon := int(math.Round(parallelCircumference / targetSpacing))
				if nLon < 4 {
					nLon = 4
				}

				dLonRing := 360.0 / float64(nLon)
				dEastMeters := parallelCircumference / float64(nLon)

				staggerLon := 0.0
				if (i % 2) == 1 {
					staggerLon = dLonRing * 0.5
				}

				rowSplats := make([]splat.Splat, 0, nLon)
				for j := 0; j < nLon; j++ {
					lon := cfg.LonMin + float64(j)*dLonRing + staggerLon
					if lon > 180.0 {
						lon -= 360.0
					}

					elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(lat, lon)
					height := cfg.BaseElevation + (elevNorm-cfg.SeaLevel+cfg.ElevationOffset)*cfg.ElevationScale
					posECEF := cfg.Spheroid.GeodeticToECEF(lat, lon, height)

					localDensity := tg.GetDensityMultiplier(lat, lon)
					scaleFactor := 1.0 / math.Sqrt(localDensity)
					stretchE := math.Sqrt(1.0 + slopeE*slopeE)
					stretchN := math.Sqrt(1.0 + slopeN*slopeN)

					sx := float32(0.5 * dEastMeters * cfg.OverlapFactor * scaleFactor * stretchE)
					sy := float32(0.5 * dNorthMeters * cfg.OverlapFactor * scaleFactor * stretchN)
					sz := float32(0.5 * math.Min(dEastMeters, dNorthMeters) * cfg.ThicknessFactor * scaleFactor)

					quat := cfg.Spheroid.OrientationQuaternion(lat, lon, slopeE, slopeN)
					color := tg.computeBiomeColor(lat, lon, elevNorm, slopeE, slopeN)

					rowSplats = append(rowSplats, splat.Splat{
						Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
						Scale:    [3]float32{sx, sy, sz},
						Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
						Color:    color,
					})

					if elevNorm < cfg.SeaLevel {
						seaHeight := cfg.BaseElevation + cfg.ElevationOffset*cfg.ElevationScale
						seaPosECEF := cfg.Spheroid.GeodeticToECEF(lat, lon, seaHeight)
						seaQuat := cfg.Spheroid.OrientationQuaternion(lat, lon, 0, 0)
						seaColor := tg.computeSeaSurfaceColor(lat, lon, elevNorm)
						seaSx := float32(0.5 * dEastMeters * cfg.OverlapFactor * scaleFactor)
						seaSy := float32(0.5 * dNorthMeters * cfg.OverlapFactor * scaleFactor)
						seaSz := float32(0.5 * math.Min(dEastMeters, dNorthMeters) * cfg.ThicknessFactor * scaleFactor * 0.5)

						rowSplats = append(rowSplats, splat.Splat{
							Position: [3]float32{float32(seaPosECEF.X), float32(seaPosECEF.Y), float32(seaPosECEF.Z)},
							Scale:    [3]float32{seaSx, seaSy, seaSz},
							Rotation: [4]float32{float32(seaQuat.W), float32(seaQuat.X), float32(seaQuat.Y), float32(seaQuat.Z)},
							Color:    seaColor,
						})
					}
				}
				ringSplats[i] = rowSplats
			}
		}()
	}
	wg.Wait()

	totalCount := 0
	for _, row := range ringSplats {
		totalCount += len(row)
	}
	splats := make([]splat.Splat, 0, totalCount)
	for _, row := range ringSplats {
		splats = append(splats, row...)
	}

	return splats
}

// sampleElevationAndSlopes samples elevation data and central-difference slope gradients in physical metric units (dh/dx, dh/dy).
func (tg *TerrainGenerator) sampleElevationAndSlopes(lat, lon float64) (elev, slopeE, slopeN float64) {
	delta := 0.05 // Small angular delta (in degrees) for sharp alpine slope gradient calculation
	deltaRad := delta * math.Pi / 180.0

	eCenter := tg.sampleHeight(lat, lon)
	eEast := tg.sampleHeight(lat, lon+delta)
	eWest := tg.sampleHeight(lat, lon-delta)
	eNorth := tg.sampleHeight(lat+delta, lon)
	eSouth := tg.sampleHeight(lat-delta, lon)

	a := tg.config.Spheroid.SemiMajorAxis
	latRad := lat * math.Pi / 180.0
	cosLat := math.Max(0.01, math.Abs(math.Cos(latRad)))

	// Physical horizontal distance in meters/units
	dx := 2.0 * deltaRad * a * cosLat
	dy := 2.0 * deltaRad * a

	// Physical vertical elevation change in meters/units
	dhE := (eEast - eWest) * tg.config.ElevationScale
	dhN := (eNorth - eSouth) * tg.config.ElevationScale

	slopeE = dhE / dx
	slopeN = dhN / dy

	return eCenter, slopeE, slopeN
}

// sampleHeight evaluates elevation in [0, 1] from high-resolution GeoTIFF LiDAR DEMs or the global heightmap.
func (tg *TerrainGenerator) sampleHeight(lat, lon float64) float64 {
	sea := tg.config.SeaLevel

	// 1. High-Resolution GeoTIFF LiDAR DEM layer (e.g. 1-meter USGS King County)
	if tg.geoTIFF != nil {
		if elevMeters, ok := tg.geoTIFF.Sample(lat, lon); ok {
			if elevMeters >= 0 {
				// Above sea level: map [0, 8848m] -> [seaLevel, 1.0]
				hNorm := sea + (elevMeters/8848.0)*(1.0-sea)
				return clampFloat(hNorm, 0.0, 1.0)
			} else {
				// Below sea level: map [-11000m, 0m] -> [0.0, seaLevel]
				hNorm := sea + (elevMeters/11000.0)*sea
				return clampFloat(hNorm, 0.0, 1.0)
			}
		}
	}

	// 2. High-Resolution Global 1km GEBCO Mosaic layer (43,200 x 21,600)
	if tg.gebco != nil {
		elev, isLand := tg.gebco.Sample(sea, lat, lon)
		if isLand {
			return elev
		}
		// In ocean areas, query raster layer for detailed ocean floor bathymetry if available
		if tg.raster != nil {
			rElev := tg.raster.Sample(lat, lon)
			if rElev < sea {
				return rElev
			}
		}
		return elev
	}

	// 3. Global DEM heightmap layer (21,600 x 10,800 fallback)
	if tg.raster != nil {
		return tg.raster.Sample(lat, lon)
	}

	// Default baseline elevation if no raster loaded
	return 0.5607
}

// computeBiomeColor maps geodetic latitude, longitude, elevation, and slope gradients to satellite-realistic terrain colors.
func (tg *TerrainGenerator) computeBiomeColor(lat, lon, elev, slopeE, slopeN float64) [4]uint8 {
	return tg.config.Biome.SampleBiomeColor(tg.config.SeaLevel, lat, lon, elev, slopeE, slopeN)
}

// computeSeaSurfaceColor maps geodetic latitude, longitude, and bed depth to translucent water colors.
func (tg *TerrainGenerator) computeSeaSurfaceColor(lat, lon, bedElev float64) [4]uint8 {
	return tg.config.Biome.SampleSeaSurfaceColor(tg.config.SeaLevel, lat, lon, bedElev)
}

func lerpFloat(t, a, b float64) float64 {
	return a + t*(b-a)
}

func clampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
