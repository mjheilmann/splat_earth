package terrain

import (
	"math"
	"runtime"
	"sync"

	"github.com/myuser/gaussian-splat-app/pkg/spheroid"
	"github.com/myuser/gaussian-splat-app/pkg/splat"
)

// GeneratorConfig holds configuration for the GIS terrain stand-in generator.
type GeneratorConfig struct {
	Spheroid spheroid.Spheroid // Reference ellipsoid

	LatMin float64 // Minimum latitude in degrees (-90 to 90)
	LatMax float64 // Maximum latitude in degrees (-90 to 90)
	LonMin float64 // Minimum longitude in degrees (-180 to 180)
	LonMax float64 // Maximum longitude in degrees (-180 to 180)

	StepsLat int // Number of latitude rings from pole to pole
	StepsLon int // Equator reference resolution for longitude

	ElevationScale  float64 // Max elevation height above ellipsoid (km or units)
	ElevationOffset float64 // Baseline elevation noise offset (-0.5 to 0.5 to shift land/sea ratio)
	BaseElevation   float64 // Sea level / baseline offset
	NoiseSeed       int64   // Seed for procedural terrain synthesizer
	NoiseFrequency  float64 // Spatial frequency of continents/mountains
	Octaves         int     // Noise detail octaves (e.g. 4-6)

	OverlapFactor      float64 // Splat size overlap multiplier (e.g. 1.25 - 1.4)
	ThicknessFactor    float64 // Ratio of vertical normal scale to horizontal tangent scale (e.g. 0.15)
	SeaLevel           float64 // Relative height threshold for water (0.0 - 1.0)
	AdaptiveDensity    bool    // Enable quadtree curvature-adaptive variable density
	CurvatureTolerance float64 // Height deviation threshold in km/units to subdivide quad cells (e.g. 0.15 km)
}

// DefaultConfig returns parameters for a complete oblate spheroid terrain.
func DefaultConfig() GeneratorConfig {
	return GeneratorConfig{
		Spheroid:           spheroid.NewSpheroid(100.0, 0.05), // Scaled spheroid: radius 100, 5% flattening
		LatMin:             -90.0,
		LatMax:             90.0,
		LonMin:             -180.0,
		LonMax:             180.0,
		StepsLat:           150,
		StepsLon:           300,
		ElevationScale:     12.0, // High relief mountains up to 12 km
		ElevationOffset:    0.0,
		BaseElevation:      0.0,
		NoiseSeed:          42,
		NoiseFrequency:     1.8,
		Octaves:            6,
		OverlapFactor:      1.35,
		ThicknessFactor:    0.035, // Wafer-thin planar discs (~3.5% of cell width) to preserve sharp topography
		SeaLevel:           0.30,
		AdaptiveDensity:    true, // Enabled by default for massive scalability
		CurvatureTolerance: 0.12, // 120 meters height deviation tolerance
	}
}

// TerrainGenerator generates 3D Gaussian Splats over an oblate spheroid.
type TerrainGenerator struct {
	config GeneratorConfig
	noise  *PerlinNoise
}

// NewTerrainGenerator initializes a new terrain generator.
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
	if cfg.Octaves < 1 {
		cfg.Octaves = 4
	}

	return &TerrainGenerator{
		config: cfg,
		noise:  NewPerlinNoise(cfg.NoiseSeed),
	}
}

// Generate creates a slice of 3D Gaussian Splats using Equal-Area adaptive sampling.
// This ensures uniform spatial resolution across the entire spheroid, eliminating polar bunching
// and concentric ring artifacts.
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
					height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale
					posECEF := cfg.Spheroid.GeodeticToECEF(lat, 0, height)

					capRadius := float32(0.5 * targetSpacing * cfg.OverlapFactor * 1.2)
					quat := cfg.Spheroid.OrientationQuaternion(lat, 0, slopeE*0.1, slopeN*0.1)
					color := tg.computeBiomeColor(elevNorm+cfg.ElevationOffset, slopeE, slopeN)

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
					height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale
					posECEF := cfg.Spheroid.GeodeticToECEF(lat, lon, height)

					sx := float32(0.5 * dEastMeters * cfg.OverlapFactor)
					sy := float32(0.5 * dNorthMeters * cfg.OverlapFactor)
					sz := float32(0.5 * math.Min(dEastMeters, dNorthMeters) * cfg.ThicknessFactor)

					quat := cfg.Spheroid.OrientationQuaternion(lat, lon, slopeE, slopeN)
					color := tg.computeBiomeColor(elevNorm, slopeE, slopeN)

					rowSplats = append(rowSplats, splat.Splat{
						Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
						Scale:    [3]float32{sx, sy, sz},
						Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
						Color:    color,
					})
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

// sampleElevationAndSlopes samples elevation noise and central-difference slope gradients in physical metric units (dh/dx, dh/dy).
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

// sampleHeight evaluates fractal elevation in [0, 1] using spherical coordinates to avoid seam distortion.
func (tg *TerrainGenerator) sampleHeight(lat, lon float64) float64 {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	// 3D coordinates on unit sphere for seamless planetary sampling
	nx := math.Cos(latRad) * math.Cos(lonRad)
	ny := math.Cos(latRad) * math.Sin(lonRad)
	nz := math.Sin(latRad)

	// 1. Crustal Domain Warping: Creates realistic curved tectonic belts, archipelagos & fjords
	wx, wy, wz := tg.noise.DomainWarp(nx, ny, nz, 0.28)

	freq := tg.config.NoiseFrequency

	// 2. Continental Plate Macro-Topology
	continental := tg.noise.FBM(wx*freq, wy*freq, wz*freq, 4, 0.52, 2.02)
	continental = (continental + 1.0) * 0.5 // Map to [0, 1]

	var height float64
	sea := tg.config.SeaLevel

	if continental > sea {
		// Landmass
		upland := (continental - sea) / (1.0 - sea) // [0, 1] relative land elevation

		// 3. Alpine Mountain Ranges: Sharp ridge noise & knife-edge peaks
		alpineRange := tg.noise.AlpineRidgeNoise(wx*freq*2.8+11.3, wy*freq*2.8+5.7, wz*freq*2.8+19.1, 4, 0.58, 2.15)
		secondaryRidge := tg.noise.RidgeNoise(nx*freq*5.2+3.1, ny*freq*5.2+7.9, nz*freq*5.2+1.4, 4, 0.50, 2.1)

		// 4. Glacial Valleys / Canyons (carved negative troughs)
		canyonNoise := 1.0 - math.Abs(tg.noise.Noise3D(nx*freq*4.0+13.7, ny*freq*4.0+8.1, nz*freq*4.0+22.4))
		canyonTrough := canyonNoise * canyonNoise * 0.12

		// 5. Rolling Foothills & Hills
		foothills := tg.noise.BillowNoise(nx*freq*3.5, ny*freq*3.5, nz*freq*3.5, 3, 0.48, 2.0)

		// 6. Close-Up Micro-Relief (boulders, cliffs, micro-ridges)
		microRelief := tg.noise.Noise3D(nx*freq*18.0, ny*freq*18.0, nz*freq*18.0) * 0.025

		// Combine layers with upland weighting (mountains rise highest in interior continental highlands)
		mountainHeight := (alpineRange*0.75 + secondaryRidge*0.25) * math.Pow(upland, 0.8) * 1.15
		height = sea + upland*0.35 + mountainHeight + foothills*0.12 - canyonTrough + microRelief
	} else {
		// Ocean Basin: Deep trenches and continental shelves
		oceanDepth := continental / sea // [0, 1] from abyssal deep to shore
		trenchNoise := tg.noise.RidgeNoise(nx*freq*2.0, ny*freq*2.0, nz*freq*2.0, 3, 0.5, 2.0) * 0.08
		height = oceanDepth*sea - trenchNoise
	}

	return clampFloat(height, 0.0, 1.0)
}

// computeBiomeColor maps elevation and slope to terrain colors.
func (tg *TerrainGenerator) computeBiomeColor(elev, slopeE, slopeN float64) [4]uint8 {
	elev = clampFloat(elev, 0.0, 1.0)
	slopeMag := math.Sqrt(slopeE*slopeE + slopeN*slopeN)
	sea := tg.config.SeaLevel

	var r, g, b float64
	alpha := uint8(255)

	if elev < sea*0.65 {
		// Deep Abyssal Trench
		t := elev / (sea * 0.65)
		r = lerpFloat(t, 12, 20)
		g = lerpFloat(t, 35, 65)
		b = lerpFloat(t, 85, 140)
	} else if elev < sea {
		// Shallow Coastal / Coral Shelf
		t := (elev - sea*0.65) / (sea * 0.35)
		r = lerpFloat(t, 20, 40)
		g = lerpFloat(t, 65, 145)
		b = lerpFloat(t, 140, 185)
	} else if elev < sea+0.035 {
		// Sandy Beaches & Dunes
		r, g, b = 218, 202, 150
	} else if elev < sea+0.22 {
		// Lush Lowland Plains & Forests
		t := (elev - (sea + 0.035)) / 0.185
		r = lerpFloat(t, 45, 65)
		g = lerpFloat(t, 145, 125)
		b = lerpFloat(t, 45, 40)
	} else if elev < sea+0.42 {
		// Subalpine Woodlands & Foothills
		t := (elev - (sea + 0.22)) / 0.20
		r = lerpFloat(t, 65, 95)
		g = lerpFloat(t, 125, 110)
		b = lerpFloat(t, 40, 65)
	} else if elev < sea+0.62 {
		// Craggy Mountain Slopes & Scree
		t := (elev - (sea + 0.42)) / 0.20
		r = lerpFloat(t, 95, 140)
		g = lerpFloat(t, 110, 135)
		b = lerpFloat(t, 65, 130)
	} else {
		// Glacial Snowcaps, Horns & Icefields
		t := (elev - (sea + 0.62)) / (1.0 - (sea + 0.62))
		if t > 1.0 {
			t = 1.0
		}
		r = lerpFloat(t, 140, 248)
		g = lerpFloat(t, 135, 250)
		b = lerpFloat(t, 130, 255)
	}

	// Steep cliff rock shading (slopes > ~15-20 degrees, slopeMag > 0.3)
	if elev > sea && slopeMag > 0.30 {
		rockWeight := math.Min((slopeMag-0.30)*1.8, 0.90)
		r = lerpFloat(rockWeight, r, 75)
		g = lerpFloat(rockWeight, g, 72)
		b = lerpFloat(rockWeight, b, 70)
	}

	return [4]uint8{uint8(clampFloat(r, 0, 255)), uint8(clampFloat(g, 0, 255)), uint8(clampFloat(b, 0, 255)), alpha}
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
