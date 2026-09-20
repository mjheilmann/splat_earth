package terrain

import (
	"math"

	"github.com/myuser/gaussian-splat-app/pkg/spheroid"
	"github.com/myuser/gaussian-splat-app/pkg/splat"
)

// Chunk represents a spatial quad tile on the oblate spheroid with multi-resolution LODs.
type Chunk struct {
	ID     int        `json:"id"`
	Row    int        `json:"row"`
	Col    int        `json:"col"`
	LatMin float64    `json:"latMin"`
	LatMax float64    `json:"latMax"`
	LonMin float64    `json:"lonMin"`
	LonMax float64    `json:"lonMax"`
	Center [3]float32 `json:"center"` // Bounding sphere center (ECEF)
	Radius float32    `json:"radius"` // Bounding sphere radius
	Normal [3]float32 `json:"normal"` // Average unit surface normal for horizon culling

	// Splats at each Level of Detail (LOD 0 = 1x, LOD 1 = 2x, LOD 2 = 4x, LOD 3 = 8x, LOD 4 = 16x downsample)
	LODs [5][]splat.Splat `json:"-"`
}

// ChunkManifest contains metadata sent to the client viewer for CPU/GPU frustum & horizon culling.
type ChunkManifest struct {
	ID        int        `json:"id"`
	Center    [3]float32 `json:"center"`
	Radius    float32    `json:"radius"`
	Normal    [3]float32 `json:"normal"`
	CountsLOD [5]int     `json:"countsLOD"`
}

// ChunkedTerrain holds all spatial chunks and aggregate metrics.
type ChunkedTerrain struct {
	Chunks    []Chunk
	Manifests []ChunkManifest
	Spheroid  spheroid.Spheroid
}

// GenerateChunkedTerrain creates a spatial grid of chunks (e.g. 8 lat divisions x 16 lon divisions)
// with hierarchical LOD pyramids for each chunk.
func (tg *TerrainGenerator) GenerateChunkedTerrain(chunkRows, chunkCols int) *ChunkedTerrain {
	if chunkRows < 1 {
		chunkRows = 8
	}
	if chunkCols < 1 {
		chunkCols = 16
	}

	cfg := tg.config
	dLatChunk := (cfg.LatMax - cfg.LatMin) / float64(chunkRows)
	dLonChunk := (cfg.LonMax - cfg.LonMin) / float64(chunkCols)

	// Step sizes per chunk
	stepsLatPerChunk := cfg.StepsLat / chunkRows
	if stepsLatPerChunk < 4 {
		stepsLatPerChunk = 4
	}
	stepsLonEquatorPerChunk := cfg.StepsLon / chunkCols
	if stepsLonEquatorPerChunk < 4 {
		stepsLonEquatorPerChunk = 4
	}

	chunks := make([]Chunk, 0, chunkRows*chunkCols)
	manifests := make([]ChunkManifest, 0, chunkRows*chunkCols)
	chunkID := 0

	for r := 0; r < chunkRows; r++ {
		latMin := cfg.LatMin + float64(r)*dLatChunk
		latMax := latMin + dLatChunk

		for c := 0; c < chunkCols; c++ {
			lonMin := cfg.LonMin + float64(c)*dLonChunk
			lonMax := lonMin + dLonChunk

			// Create sub-generator config for this chunk
			chunkCfg := cfg
			chunkCfg.LatMin = latMin
			chunkCfg.LatMax = latMax
			chunkCfg.LonMin = lonMin
			chunkCfg.LonMax = lonMax
			chunkCfg.StepsLat = stepsLatPerChunk
			chunkCfg.StepsLon = stepsLonEquatorPerChunk * chunkCols

			chunkGen := NewTerrainGenerator(chunkCfg)

			// 1. Generate LOD 0 (Full resolution)
			lod0 := chunkGen.generateChunkSplats(1.0, 1)

			// 2. Generate LOD 1 (2x downsampled)
			lod1 := chunkGen.generateChunkSplats(2.0, 2)

			// 3. Generate LOD 2 (4x downsampled)
			lod2 := chunkGen.generateChunkSplats(4.0, 4)

			// 4. Generate LOD 3 (8x downsampled)
			lod3 := chunkGen.generateChunkSplats(8.0, 8)

			// 5. Generate LOD 4 (16x downsampled, deep orbit LOD)
			lod4 := chunkGen.generateChunkSplats(16.0, 16)

			// Compute Bounding Sphere and Surface Normal for this chunk
			center, radius, normal := computeBoundingSphereAndNormal(lod0, cfg.Spheroid, (latMin+latMax)*0.5, (lonMin+lonMax)*0.5)

			chunk := Chunk{
				ID:     chunkID,
				Row:    r,
				Col:    c,
				LatMin: latMin,
				LatMax: latMax,
				LonMin: lonMin,
				LonMax: lonMax,
				Center: center,
				Radius: radius,
				Normal: normal,
				LODs:   [5][]splat.Splat{lod0, lod1, lod2, lod3, lod4},
			}

			chunks = append(chunks, chunk)
			manifests = append(manifests, ChunkManifest{
				ID:        chunkID,
				Center:    center,
				Radius:    radius,
				Normal:    normal,
				CountsLOD: [5]int{len(lod0), len(lod1), len(lod2), len(lod3), len(lod4)},
			})

			chunkID++
		}
	}

	return &ChunkedTerrain{
		Chunks:    chunks,
		Manifests: manifests,
		Spheroid:  cfg.Spheroid,
	}
}

// generateChunkSplats generates splats for a specific chunk with downsampling stride and footprint bloat factor.
func (tg *TerrainGenerator) generateChunkSplats(footprintScale float64, stride int) []splat.Splat {
	cfg := tg.config
	splats := make([]splat.Splat, 0, (cfg.StepsLat/stride)*(cfg.StepsLon/stride)+2)

	dLat := (cfg.LatMax - cfg.LatMin) / float64(cfg.StepsLat-1)
	dLatRad := dLat * math.Pi / 180.0

	targetSpacing := ((2.0 * math.Pi * cfg.Spheroid.SemiMajorAxis) / float64(cfg.StepsLon)) * float64(stride)

	for i := 0; i < cfg.StepsLat; i += stride {
		lat := cfg.LatMin + float64(i)*dLat
		latRad := lat * math.Pi / 180.0

		// Handle poles
		if math.Abs(lat) >= 89.99 {
			elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(lat, 0)
			height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale
			posECEF := cfg.Spheroid.GeodeticToECEF(lat, 0, height)

			capRadius := float32(0.5 * targetSpacing * cfg.OverlapFactor * footprintScale * 1.2)
			capThickness := float32(0.5 * targetSpacing * cfg.ThicknessFactor)
			quat := cfg.Spheroid.OrientationQuaternion(lat, 0, slopeE*0.1, slopeN*0.1)
			color := tg.computeBiomeColor(elevNorm, slopeE, slopeN)

			splats = append(splats, splat.Splat{
				Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
				Scale:    [3]float32{capRadius, capRadius, capThickness},
				Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
				Color:    color,
			})
			continue
		}

		sinLat := math.Sin(latRad)
		cosLat := math.Cos(latRad)

		denom := math.Sqrt(1.0 - cfg.Spheroid.Eccentricity2*sinLat*sinLat)
		N := cfg.Spheroid.SemiMajorAxis / denom
		M := cfg.Spheroid.SemiMajorAxis * (1.0 - cfg.Spheroid.Eccentricity2) / (denom * denom * denom)

		dNorthMeters := M * dLatRad * float64(stride)
		parallelCircumference := 2.0 * math.Pi * N * math.Abs(cosLat)

		// Equal-Area Adaptive Sampling: Number of points on this ring proportional to parallel radius
		nLon := int(math.Round(parallelCircumference / targetSpacing))
		if nLon < 2 {
			nLon = 2
		}

		// Fractional chunk coverage of longitude
		chunkLonFraction := (cfg.LonMax - cfg.LonMin) / 360.0
		nLonChunk := int(math.Max(1, math.Round(float64(nLon)*chunkLonFraction)))

		dLonRing := (cfg.LonMax - cfg.LonMin) / float64(nLonChunk)
		dEastMeters := (parallelCircumference / float64(nLon)) * float64(stride)

		for j := 0; j < nLonChunk; j++ {
			lon := cfg.LonMin + (float64(j)+0.5)*dLonRing

			// Sample procedural elevation
			elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(lat, lon)
			height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale

			// Compute 3D ECEF position
			posECEF := cfg.Spheroid.GeodeticToECEF(lat, lon, height)

			// Bloated footprint scale for lower LODs to prevent pinholes
			sx := float32(0.5 * dEastMeters * cfg.OverlapFactor * footprintScale)
			sy := float32(0.5 * dNorthMeters * cfg.OverlapFactor * footprintScale)
			sz := float32(0.5 * math.Min(dEastMeters, dNorthMeters) * cfg.ThicknessFactor)

			quat := cfg.Spheroid.OrientationQuaternion(lat, lon, slopeE*0.1, slopeN*0.1)
			color := tg.computeBiomeColor(elevNorm, slopeE, slopeN)

			splats = append(splats, splat.Splat{
				Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
				Scale:    [3]float32{sx, sy, sz},
				Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
				Color:    color,
			})
		}
	}

	return splats
}

// computeBoundingSphereAndNormal calculates the minimal bounding sphere center, radius, and average normal for a chunk.
func computeBoundingSphereAndNormal(splats []splat.Splat, s spheroid.Spheroid, midLat, midLon float64) (center [3]float32, radius float32, normal [3]float32) {
	if len(splats) == 0 {
		pos := s.GeodeticToECEF(midLat, midLon, 0)
		n := s.SurfaceNormal(midLat, midLon)
		return [3]float32{float32(pos.X), float32(pos.Y), float32(pos.Z)}, 10.0, [3]float32{float32(n.X), float32(n.Y), float32(n.Z)}
	}

	var sumX, sumY, sumZ float64
	for _, p := range splats {
		sumX += float64(p.Position[0])
		sumY += float64(p.Position[1])
		sumZ += float64(p.Position[2])
	}
	n := float64(len(splats))
	cX, cY, cZ := sumX/n, sumY/n, sumZ/n

	var maxR float64
	for _, p := range splats {
		dx := float64(p.Position[0]) - cX
		dy := float64(p.Position[1]) - cY
		dz := float64(p.Position[2]) - cZ
		r := math.Sqrt(dx*dx + dy*dy + dz*dz)
		if r > maxR {
			maxR = r
		}
	}

	norm := s.SurfaceNormal(midLat, midLon)

	return [3]float32{float32(cX), float32(cY), float32(cZ)},
		float32(maxR * 1.15), // Small safety padding
		[3]float32{float32(norm.X), float32(norm.Y), float32(norm.Z)}
}
