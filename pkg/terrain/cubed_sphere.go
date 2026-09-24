package terrain

import (
	"math"
	"runtime"
	"sync"

	"github.com/mjheilmann/gaussian-splat-app/pkg/spheroid"
	"github.com/mjheilmann/gaussian-splat-app/pkg/splat"
)

// CubeFace represents one of the 6 cardinal faces of a spherified cube.
type CubeFace int

const (
	FacePosX CubeFace = iota // +X (0° Lon Equator)
	FacePosY                 // +Y (+90° Lon Equator)
	FaceNegX                 // -X (180° Lon Equator)
	FaceNegY                 // -Y (-90° Lon Equator)
	FacePosZ                 // +Z (North Pole Face)
	FaceNegZ                 // -Z (South Pole Face)
)

// FaceBasis returns the origin and tangent axes for a cube face in [-1, 1]^3 space.
func (f CubeFace) FaceBasis() (center, uAxis, vAxis spheroid.Vec3) {
	switch f {
	case FacePosX:
		return spheroid.Vec3{X: 1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: 1, Z: 0}, spheroid.Vec3{X: 0, Y: 0, Z: 1}
	case FacePosY:
		return spheroid.Vec3{X: 0, Y: 1, Z: 0}, spheroid.Vec3{X: -1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: 0, Z: 1}
	case FaceNegX:
		return spheroid.Vec3{X: -1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: -1, Z: 0}, spheroid.Vec3{X: 0, Y: 0, Z: 1}
	case FaceNegY:
		return spheroid.Vec3{X: 0, Y: -1, Z: 0}, spheroid.Vec3{X: 1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: 0, Z: 1}
	case FacePosZ:
		return spheroid.Vec3{X: 0, Y: 0, Z: 1}, spheroid.Vec3{X: 1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: 1, Z: 0}
	case FaceNegZ:
		return spheroid.Vec3{X: 0, Y: 0, Z: -1}, spheroid.Vec3{X: 1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: -1, Z: 0}
	}
	return spheroid.Vec3{X: 1, Y: 0, Z: 0}, spheroid.Vec3{X: 0, Y: 1, Z: 0}, spheroid.Vec3{X: 0, Y: 0, Z: 1}
}

// DensityMetrics contains spatial density and chunk recommendations.
type DensityMetrics struct {
	TotalPoints               int     `json:"totalPoints"`
	SurfaceAreaKm2            float64 `json:"surfaceAreaKm2"`
	SplatsPerKm2              float64 `json:"splatsPerKm2"`
	MeanSpacingMeters         float64 `json:"meanSpacingMeters"`
	RecommendedPatchesPerFace int     `json:"recommendedPatchesPerFace"`
	TotalChunks               int     `json:"totalChunks"`
	SplatsPerChunkLOD0        int     `json:"splatsPerChunkLOD0"`
	FocusEnabled              bool    `json:"focusEnabled"`
	FocusSplatsPerKm2         float64 `json:"focusSplatsPerKm2,omitempty"`
	FocusMeanSpacingMeters    float64 `json:"focusMeanSpacingMeters,omitempty"`
}

// EstimatePointDensity calculates the spatial density and recommends optimal chunk partitioning.
func (tg *TerrainGenerator) EstimatePointDensity(totalPoints int) DensityMetrics {
	if totalPoints <= 0 {
		totalPoints = tg.config.StepsLat * tg.config.StepsLon
	}
	a := tg.config.Spheroid.SemiMajorAxis
	b := tg.config.Spheroid.SemiMinorAxis

	e := math.Sqrt(math.Max(0.0, 1.0-(b*b)/(a*a)))
	var surfaceArea float64
	if e < 1e-4 {
		surfaceArea = 4.0 * math.Pi * a * a
	} else {
		surfaceArea = 2.0*math.Pi*a*a + math.Pi*(b*b/e)*math.Log((1.0+e)/(1.0-e))
	}

	splatsPerKm2 := float64(totalPoints) / surfaceArea
	meanSpacingKm := math.Sqrt(surfaceArea / float64(totalPoints))
	meanSpacingMeters := meanSpacingKm * 1000.0

	// Target ~4,000 to ~12,000 splats per chunk at LOD 0 for optimal GPU streaming & culling
	targetChunks := float64(totalPoints) / 8192.0
	patchesPerFace := int(math.Round(math.Sqrt(targetChunks / 6.0)))
	if patchesPerFace < 2 {
		patchesPerFace = 2 // Minimum 6 x 4 = 24 chunks
	} else if patchesPerFace > 32 {
		patchesPerFace = 32 // Maximum 6 x 1024 = 6144 chunks
	}

	totalChunksCount := 6 * patchesPerFace * patchesPerFace
	splatsPerChunk := totalPoints / totalChunksCount

	metrics := DensityMetrics{
		TotalPoints:               totalPoints,
		SurfaceAreaKm2:            surfaceArea,
		SplatsPerKm2:              splatsPerKm2,
		MeanSpacingMeters:         meanSpacingMeters,
		RecommendedPatchesPerFace: patchesPerFace,
		TotalChunks:               totalChunksCount,
		SplatsPerChunkLOD0:        splatsPerChunk,
	}

	activeZones := tg.config.ActiveFocusAreas()
	if len(activeZones) > 0 {
		metrics.FocusEnabled = true
		primary := activeZones[0]
		mult := primary.DensityMultiplier
		if mult < 1.0 {
			mult = 1.0
		}
		metrics.FocusSplatsPerKm2 = splatsPerKm2 * mult
		metrics.FocusMeanSpacingMeters = meanSpacingMeters / math.Sqrt(mult)
	}

	return metrics
}

// lodTargetResolutions defines the target point count per patch axis for each LOD level.
// LOD 0 targets max 192x192 (~36k splats/chunk) so close-up ground flight remains rock-solid and never exhausts browser memory.
// LOD 1-4 target fixed downsampled budgets so high-altitude and orbit rendering remain under 60k splats globally.
var lodTargetResolutions = [5]int{192, 96, 48, 24, 12}

func computeLODStride(nativeRes int, lod int) int {
	if lod <= 0 {
		if nativeRes <= 192 {
			return 1
		}
		return int(math.Max(1, math.Round(float64(nativeRes)/192.0)))
	}
	if lod > 4 {
		lod = 4
	}
	targetRes := lodTargetResolutions[lod]
	stride := 1 << lod
	if targetRes > 0 && nativeRes > targetRes {
		targetStride := int(math.Round(float64(nativeRes) / float64(targetRes)))
		if targetStride > stride {
			stride = targetStride
		}
	}
	if stride > nativeRes && nativeRes > 0 {
		stride = nativeRes
	}
	if stride < 1 {
		stride = 1
	}
	return stride
}

// GenerateCubedSphereTerrain partitions the oblate spheroid into 6 cube faces,
// each split into patchesPerFaceAxis x patchesPerFaceAxis spatial chunks with multi-LOD splats.
// If patchesPerFaceAxis <= 0, it automatically estimates point density and selects the optimal chunk grid.
// Chunk generation is fully parallelized across all available CPU cores.
func (tg *TerrainGenerator) GenerateCubedSphereTerrain(patchesPerFaceAxis int) *ChunkedTerrain {
	if patchesPerFaceAxis <= 0 {
		metrics := tg.EstimatePointDensity(0)
		patchesPerFaceAxis = metrics.RecommendedPatchesPerFace
	}

	cfg := tg.config
	totalChunks := 6 * patchesPerFaceAxis * patchesPerFaceAxis
	chunks := make([]Chunk, totalChunks)
	manifests := make([]ChunkManifest, totalChunks)

	// Total points target divided evenly across all 6 faces and patches
	totalPointsTarget := cfg.StepsLat * cfg.StepsLon
	pointsPerPatchSide := int(math.Max(4, math.Round(math.Sqrt(float64(totalPointsTarget)/float64(totalChunks)))))

	type patchTask struct {
		chunkID int
		face    int
		pu      int
		pv      int
		uMin    float64
		uMax    float64
		vMin    float64
		vMax    float64
	}

	tasks := make(chan patchTask, totalChunks)
	patchSize := 2.0 / float64(patchesPerFaceAxis)
	id := 0

	for face := 0; face < 6; face++ {
		for pu := 0; pu < patchesPerFaceAxis; pu++ {
			uMin := -1.0 + float64(pu)*patchSize
			uMax := uMin + patchSize

			for pv := 0; pv < patchesPerFaceAxis; pv++ {
				vMin := -1.0 + float64(pv)*patchSize
				vMax := vMin + patchSize

				tasks <- patchTask{
					chunkID: id,
					face:    face,
					pu:      pu,
					pv:      pv,
					uMin:    uMin,
					uMax:    uMax,
					vMin:    vMin,
					vMax:    vMax,
				}
				id++
			}
		}
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
				cubeFace := CubeFace(task.face)
				fCenter, fUAxis, fVAxis := cubeFace.FaceBasis()

				s0 := computeLODStride(pointsPerPatchSide, 0)
				s1 := computeLODStride(pointsPerPatchSide, 1)
				s2 := computeLODStride(pointsPerPatchSide, 2)
				s3 := computeLODStride(pointsPerPatchSide, 3)
				s4 := computeLODStride(pointsPerPatchSide, 4)

				lod0 := tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, task.uMin, task.uMax, task.vMin, task.vMax, pointsPerPatchSide, patchesPerFaceAxis, s0)
				lod1 := tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, task.uMin, task.uMax, task.vMin, task.vMax, pointsPerPatchSide, patchesPerFaceAxis, s1)
				lod2 := tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, task.uMin, task.uMax, task.vMin, task.vMax, pointsPerPatchSide, patchesPerFaceAxis, s2)
				lod3 := tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, task.uMin, task.uMax, task.vMin, task.vMax, pointsPerPatchSide, patchesPerFaceAxis, s3)
				lod4 := tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, task.uMin, task.uMax, task.vMin, task.vMax, pointsPerPatchSide, patchesPerFaceAxis, s4)

				center, radius, normal := computePatchBounds(lod0, cfg.Spheroid, fCenter, fUAxis, fVAxis, (task.uMin+task.uMax)*0.5, (task.vMin+task.vMax)*0.5)

				chunk := Chunk{
					ID:     task.chunkID,
					Row:    task.face*patchesPerFaceAxis + task.pu,
					Col:    task.pv,
					Center: center,
					Radius: radius,
					Normal: normal,
					LODs:   [5][]splat.Splat{lod0, lod1, lod2, lod3, lod4},
				}

				chunks[task.chunkID] = chunk
				manifests[task.chunkID] = ChunkManifest{
					ID:        task.chunkID,
					Center:    center,
					Radius:    radius,
					Normal:    normal,
					CountsLOD: [5]int{len(lod0), len(lod1), len(lod2), len(lod3), len(lod4)},
				}
			}
		}()
	}
	wg.Wait()

	return &ChunkedTerrain{
		Chunks:    chunks,
		Manifests: manifests,
		Spheroid:  cfg.Spheroid,
	}
}

// ProceduralCubedSphere provides O(1) instant chunk manifest generation and on-demand streaming for billions of splats with zero memory paging.
type ProceduralCubedSphere struct {
	Config             GeneratorConfig
	PatchesPerFaceAxis int
	TotalChunks        int
	PointsPerPatchSide int
	Manifests          []ChunkManifest
}

// NewProceduralCubedSphere computes spatial chunk bounds in O(1) without allocating splat slices in RAM.
func (tg *TerrainGenerator) NewProceduralCubedSphere(patchesPerFaceAxis int) *ProceduralCubedSphere {
	if patchesPerFaceAxis <= 0 {
		metrics := tg.EstimatePointDensity(0)
		patchesPerFaceAxis = metrics.RecommendedPatchesPerFace
	}

	cfg := tg.config
	totalChunks := 6 * patchesPerFaceAxis * patchesPerFaceAxis
	manifests := make([]ChunkManifest, totalChunks)

	totalPointsTarget := cfg.StepsLat * cfg.StepsLon
	pointsPerPatchSide := int(math.Max(4, math.Round(math.Sqrt(float64(totalPointsTarget)/float64(totalChunks)))))
	patchSize := 2.0 / float64(patchesPerFaceAxis)

	id := 0
	for face := 0; face < 6; face++ {
		cubeFace := CubeFace(face)
		fCenter, fUAxis, fVAxis := cubeFace.FaceBasis()

		for pu := 0; pu < patchesPerFaceAxis; pu++ {
			uMin := -1.0 + float64(pu)*patchSize
			uMax := uMin + patchSize
			midU := (uMin + uMax) * 0.5

			for pv := 0; pv < patchesPerFaceAxis; pv++ {
				vMin := -1.0 + float64(pv)*patchSize
				vMax := vMin + patchSize
				midV := (vMin + vMax) * 0.5

				// Analytical chunk center and normal
				cX := fCenter.X + midU*fUAxis.X + midV*fVAxis.X
				cY := fCenter.Y + midU*fUAxis.Y + midV*fVAxis.Y
				cZ := fCenter.Z + midU*fUAxis.Z + midV*fVAxis.Z
				lenC := math.Sqrt(cX*cX + cY*cY + cZ*cZ)
				if lenC < 1e-6 {
					lenC = 1.0
				}
				norm := [3]float32{float32(cX / lenC), float32(cY / lenC), float32(cZ / lenC)}

				latRad := math.Asin(math.Max(-1.0, math.Min(1.0, cZ/lenC)))
				lonRad := math.Atan2(cY, cX)
				centerPos := cfg.Spheroid.GeodeticToECEF(latRad*180/math.Pi, lonRad*180/math.Pi, cfg.ElevationScale*0.5)
				center := [3]float32{float32(centerPos.X), float32(centerPos.Y), float32(centerPos.Z)}

				// Analytical bounding radius
				cornerX := fCenter.X + uMax*fUAxis.X + vMax*fVAxis.X
				cornerY := fCenter.Y + uMax*fUAxis.Y + vMax*fVAxis.Y
				cornerZ := fCenter.Z + uMax*fUAxis.Z + vMax*fVAxis.Z
				lenCorner := math.Sqrt(cornerX*cornerX + cornerY*cornerY + cornerZ*cornerZ)
				cornerLat := math.Asin(math.Max(-1.0, math.Min(1.0, cornerZ/lenCorner))) * 180 / math.Pi
				cornerLon := math.Atan2(cornerY, cornerX) * 180 / math.Pi
				cornerPos := cfg.Spheroid.GeodeticToECEF(cornerLat, cornerLon, cfg.ElevationScale)
				dx := float64(center[0]) - cornerPos.X
				dy := float64(center[1]) - cornerPos.Y
				dz := float64(center[2]) - cornerPos.Z
				radius := float32(math.Sqrt(dx*dx+dy*dy+dz*dz)*1.25 + 2.0)

				s0 := computeLODStride(pointsPerPatchSide, 0)
				s1 := computeLODStride(pointsPerPatchSide, 1)
				s2 := computeLODStride(pointsPerPatchSide, 2)
				s3 := computeLODStride(pointsPerPatchSide, 3)
				s4 := computeLODStride(pointsPerPatchSide, 4)

				n0 := (pointsPerPatchSide + s0 - 1) / s0
				n1 := (pointsPerPatchSide + s1 - 1) / s1
				n2 := (pointsPerPatchSide + s2 - 1) / s2
				n3 := (pointsPerPatchSide + s3 - 1) / s3
				n4 := (pointsPerPatchSide + s4 - 1) / s4

				c0 := n0 * n0
				c1 := n1 * n1
				c2 := n2 * n2
				c3 := n3 * n3
				c4 := n4 * n4

				chunkDensity := tg.GetDensityMultiplier(latRad*180/math.Pi, lonRad*180/math.Pi)
				if chunkDensity > 1.05 {
					c0 = int(float64(c0) * chunkDensity)
				}

				if c0 < 1 { c0 = 1 }
				if c1 < 1 { c1 = 1 }
				if c2 < 1 { c2 = 1 }
				if c3 < 1 { c3 = 1 }
				if c4 < 1 { c4 = 1 }

				manifests[id] = ChunkManifest{
					ID:        id,
					Center:    center,
					Radius:    radius,
					Normal:    norm,
					CountsLOD: [5]int{c0, c1, c2, c3, c4},
				}
				id++
			}
		}
	}

	return &ProceduralCubedSphere{
		Config:             cfg,
		PatchesPerFaceAxis: patchesPerFaceAxis,
		TotalChunks:        totalChunks,
		PointsPerPatchSide: pointsPerPatchSide,
		Manifests:          manifests,
	}
}

// SampleChunkLOD generates splats on demand for a single requested chunk and LOD.
func (p *ProceduralCubedSphere) SampleChunkLOD(chunkID int, lod int) []splat.Splat {
	if chunkID < 0 || chunkID >= p.TotalChunks {
		return nil
	}

	patchesPerFace := p.PatchesPerFaceAxis
	face := chunkID / (patchesPerFace * patchesPerFace)
	rem := chunkID % (patchesPerFace * patchesPerFace)
	pu := rem / patchesPerFace
	pv := rem % patchesPerFace

	patchSize := 2.0 / float64(patchesPerFace)
	uMin := -1.0 + float64(pu)*patchSize
	uMax := uMin + patchSize
	vMin := -1.0 + float64(pv)*patchSize
	vMax := vMin + patchSize

	cubeFace := CubeFace(face)
	fCenter, fUAxis, fVAxis := cubeFace.FaceBasis()

	stride := computeLODStride(p.PointsPerPatchSide, lod)

	tg := NewTerrainGenerator(p.Config)
	return tg.sampleCubedSpherePatch(cubeFace, fCenter, fUAxis, fVAxis, uMin, uMax, vMin, vMax, p.PointsPerPatchSide, patchesPerFace, stride)
}

// GenerateCubedSphereSplats generates a single combined slice of LOD 0 splats across all 6 faces of the cubed spheroid.
func (tg *TerrainGenerator) GenerateCubedSphereSplats(patchesPerFaceAxis int) []splat.Splat {
	chunked := tg.GenerateCubedSphereTerrain(patchesPerFaceAxis)
	total := 0
	for _, ch := range chunked.Chunks {
		total += len(ch.LODs[0])
	}
	all := make([]splat.Splat, 0, total)
	for _, ch := range chunked.Chunks {
		all = append(all, ch.LODs[0]...)
	}
	return all
}

// sampleCubedSpherePatch samples Gaussian splats over a spherified cube face patch.
// Subdivides to baseDepth across the general planet, and to maxFocusDepth in steered high-density focus areas.
func (tg *TerrainGenerator) sampleCubedSpherePatch(
	face CubeFace, fCenter, fUAxis, fVAxis spheroid.Vec3,
	uMin, uMax, vMin, vMax float64,
	res int,
	patchesPerFaceAxis int,
	stride int,
) []splat.Splat {
	cfg := tg.config

	// Physical target metric spacing on the spheroid
	surfaceArea := 4.0 * math.Pi * cfg.Spheroid.SemiMajorAxis * cfg.Spheroid.SemiMajorAxis
	patchArea := surfaceArea / float64(6*patchesPerFaceAxis*patchesPerFaceAxis)

	ptsPerSide := float64(res) / float64(stride)
	if ptsPerSide < 1.0 {
		ptsPerSide = 1.0
	}
	baseDepth := int(math.Round(math.Log2(ptsPerSide)))
	if baseDepth < 1 {
		baseDepth = 1
	}

	activeZones := cfg.ActiveFocusAreas()
	hasSteeredDensity := len(activeZones) > 0

	if cfg.AdaptiveDensity || hasSteeredDensity {
		maxFocusDepth := baseDepth
		for _, z := range activeZones {
			mult := z.DensityMultiplier
			if mult < 2.0 {
				mult = 2.0
			}
			boost := int(math.Ceil(0.5 * math.Log2(mult))) + 1
			if boost < 2 {
				boost = 2
			}
			depthForZone := baseDepth + boost
			if depthForZone < 6 {
				depthForZone = 6 // Ensure crisp microscopic detail in focus zone
			}
			if depthForZone > 11 {
				depthForZone = 11
			}
			if depthForZone > maxFocusDepth {
				maxFocusDepth = depthForZone
			}
		}

		splats := make([]splat.Splat, 0, 256)
		tg.sampleAdaptiveQuad(
			face, fCenter, fUAxis, fVAxis,
			uMin, uMax, vMin, vMax,
			uMin, uMax, vMin, vMax,
			0, baseDepth, maxFocusDepth, patchArea,
			&splats,
		)
		return splats
	}

	// Uniform grid sampling fallback
	splats := make([]splat.Splat, 0, (res/stride)*(res/stride)+1)

	du := (uMax - uMin) / float64(res)
	dv := (vMax - vMin) / float64(res)

	splatSpacing := math.Sqrt(patchArea) / ptsPerSide

	for i := 0; i < res; i += stride {
		// Sample at the centroid of each downsampled grid cell for crisp alignment
		u := uMin + (float64(i)+float64(stride)*0.5)*du

		for j := 0; j < res; j += stride {
			v := vMin + (float64(j)+float64(stride)*0.5)*dv

			// Spherify cube coordinates: v_cube = center + u*uAxis + v*vAxis
			cX := fCenter.X + u*fUAxis.X + v*fVAxis.X
			cY := fCenter.Y + u*fUAxis.Y + v*fVAxis.Y
			cZ := fCenter.Z + u*fUAxis.Z + v*fVAxis.Z

			lenC := math.Sqrt(cX*cX + cY*cY + cZ*cZ)
			if lenC < 1e-6 {
				lenC = 1.0
			}
			nX := cX / lenC
			nY := cY / lenC
			nZ := cZ / lenC

			// Convert unit sphere normal to Geodetic Latitude & Longitude
			latRad := math.Asin(math.Max(-1.0, math.Min(1.0, nZ)))
			lonRad := math.Atan2(nY, nX)

			latDeg := latRad * 180.0 / math.Pi
			lonDeg := lonRad * 180.0 / math.Pi

			// Sample elevation noise and slopes
			elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(latDeg, lonDeg)
			height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale

			// Transform to Oblate Spheroid ECEF Cartesian position
			posECEF := cfg.Spheroid.GeodeticToECEF(latDeg, lonDeg, height)

			// Gaussian Splat Scale: Tangent footprint matching local density spacing
			localDensity := tg.GetDensityMultiplier(latDeg, lonDeg)
			localSpacing := splatSpacing
			if localDensity > 1.0 {
				localSpacing = splatSpacing / math.Sqrt(localDensity)
			}

			sx := float32(0.5 * localSpacing * cfg.OverlapFactor)
			sy := float32(0.5 * localSpacing * cfg.OverlapFactor)
			sz := float32(0.5 * localSpacing * cfg.ThicknessFactor)

			// Normal-aligned rotation quaternion
			quat := cfg.Spheroid.OrientationQuaternion(latDeg, lonDeg, slopeE, slopeN)

			// Biome color
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

// sampleAdaptiveQuad recursively subdivides a patch quadrant based on local terrain curvature / elevation deviation
// and steered high-density focus areas.
func (tg *TerrainGenerator) sampleAdaptiveQuad(
	face CubeFace, fCenter, fUAxis, fVAxis spheroid.Vec3,
	u0, u1, v0, v1 float64,
	patchUMin, patchUMax, patchVMin, patchVMax float64,
	depth, baseDepth, maxFocusDepth int,
	patchArea float64,
	splats *[]splat.Splat,
) {
	cfg := tg.config

	// Convert coordinates to lat/lon helper
	toLatLon := func(u, v float64) (float64, float64) {
		cX := fCenter.X + u*fUAxis.X + v*fVAxis.X
		cY := fCenter.Y + u*fUAxis.Y + v*fVAxis.Y
		cZ := fCenter.Z + u*fUAxis.Z + v*fVAxis.Z
		lenC := math.Sqrt(cX*cX + cY*cY + cZ*cZ)
		if lenC < 1e-6 {
			lenC = 1.0
		}
		latRad := math.Asin(math.Max(-1.0, math.Min(1.0, cZ/lenC)))
		lonRad := math.Atan2(cY, cX)
		return latRad * 180.0 / math.Pi, lonRad * 180.0 / math.Pi
	}

	uMid := (u0 + u1) * 0.5
	vMid := (v0 + v1) * 0.5

	latMid, lonMid := toLatLon(uMid, vMid)
	elevMid := tg.sampleHeight(latMid, lonMid)

	// Sample 4 corners
	lat00, lon00 := toLatLon(u0, v0)
	lat10, lon10 := toLatLon(u1, v0)
	lat01, lon01 := toLatLon(u0, v1)
	lat11, lon11 := toLatLon(u1, v1)

	h00 := tg.sampleHeight(lat00, lon00)
	h10 := tg.sampleHeight(lat10, lon10)
	h01 := tg.sampleHeight(lat01, lon01)
	h11 := tg.sampleHeight(lat11, lon11)

	// Bilinear center approximation
	hBilinear := (h00 + h10 + h01 + h11) * 0.25

	// Height curvature error in km/units
	errH := math.Abs(elevMid-hBilinear) * cfg.ElevationScale
	spanH := math.Max(math.Abs(h00-h11), math.Abs(h10-h01)) * cfg.ElevationScale

	tol := cfg.CurvatureTolerance
	if tol <= 0 {
		tol = 0.12 // 120m default tolerance
	}

	// Metric cell size and bounding radius of this quad on the spheroid
	cellFraction := (u1 - u0) / (patchUMax - patchUMin)
	cellSpacing := math.Sqrt(patchArea) * cellFraction
	quadRadiusKm := cellSpacing * 0.7071

	targetDepth := baseDepth
	activeZones := cfg.ActiveFocusAreas()

	for _, z := range activeZones {
		mult := z.DensityMultiplier
		if mult < 2.0 {
			mult = 2.0
		}
		boost := int(math.Ceil(0.5 * math.Log2(mult))) + 1
		if boost < 2 {
			boost = 2
		}
		zoneTargetDepth := baseDepth + boost
		if zoneTargetDepth < 6 {
			zoneTargetDepth = 6
		}
		if zoneTargetDepth > maxFocusDepth {
			zoneTargetDepth = maxFocusDepth
		}

		radius := z.RadiusKm()
		trans := z.TransitionWidthKm
		if trans <= 0 {
			trans = math.Max(0.5, radius*0.6)
		}

		dist := cfg.Spheroid.SurfaceDistance(latMid, lonMid, z.CenterLat, z.CenterLon)
		minDist := math.Max(0.0, dist-quadRadiusKm)
		maxDist := dist + quadRadiusKm

		zoneDepth := baseDepth
		if maxDist <= radius {
			// Entire quad is completely inside focus core
			zoneDepth = zoneTargetDepth
		} else if minDist <= radius {
			// Quad overlaps the focus core boundary
			zoneDepth = zoneTargetDepth
		} else if minDist < radius+trans {
			// Quad is in smooth transition margin
			t := (minDist - radius) / trans
			w := 0.5 * (1.0 + math.Cos(math.Pi*t))
			zoneDepth = baseDepth + int(math.Round(float64(zoneTargetDepth-baseDepth)*w))
		}

		if zoneDepth > targetDepth {
			targetDepth = zoneDepth
		}
	}

	shouldSubdivide := false
	if depth < targetDepth {
		shouldSubdivide = true
	} else if cfg.AdaptiveDensity && depth < maxFocusDepth {
		if errH > tol || (depth < 2 && elevMid > cfg.SeaLevel && spanH > tol*2.5) {
			shouldSubdivide = true
		}
	}

	if shouldSubdivide {
		// Subdivide into 4 child quadrants
		tg.sampleAdaptiveQuad(face, fCenter, fUAxis, fVAxis, u0, uMid, v0, vMid, patchUMin, patchUMax, patchVMin, patchVMax, depth+1, baseDepth, maxFocusDepth, patchArea, splats)
		tg.sampleAdaptiveQuad(face, fCenter, fUAxis, fVAxis, uMid, u1, v0, vMid, patchUMin, patchUMax, patchVMin, patchVMax, depth+1, baseDepth, maxFocusDepth, patchArea, splats)
		tg.sampleAdaptiveQuad(face, fCenter, fUAxis, fVAxis, u0, uMid, vMid, v1, patchUMin, patchUMax, patchVMin, patchVMax, depth+1, baseDepth, maxFocusDepth, patchArea, splats)
		tg.sampleAdaptiveQuad(face, fCenter, fUAxis, fVAxis, uMid, u1, vMid, v1, patchUMin, patchUMax, patchVMin, patchVMax, depth+1, baseDepth, maxFocusDepth, patchArea, splats)
		return
	}

	// Leaf cell: Emit single adaptive splat covering this cell
	elevNorm, slopeE, slopeN := tg.sampleElevationAndSlopes(latMid, lonMid)
	height := cfg.BaseElevation + (elevNorm+cfg.ElevationOffset)*cfg.ElevationScale
	posECEF := cfg.Spheroid.GeodeticToECEF(latMid, lonMid, height)

	sx := float32(0.5 * cellSpacing * cfg.OverlapFactor)
	sy := float32(0.5 * cellSpacing * cfg.OverlapFactor)
	sz := float32(0.5 * cellSpacing * cfg.ThicknessFactor)

	quat := cfg.Spheroid.OrientationQuaternion(latMid, lonMid, slopeE, slopeN)
	color := tg.computeBiomeColor(elevNorm, slopeE, slopeN)

	*splats = append(*splats, splat.Splat{
		Position: [3]float32{float32(posECEF.X), float32(posECEF.Y), float32(posECEF.Z)},
		Scale:    [3]float32{sx, sy, sz},
		Rotation: [4]float32{float32(quat.W), float32(quat.X), float32(quat.Y), float32(quat.Z)},
		Color:    color,
	})
}

func computePatchBounds(splats []splat.Splat, s spheroid.Spheroid, fCenter, fUAxis, fVAxis spheroid.Vec3, midU, midV float64) ([3]float32, float32, [3]float32) {
	if len(splats) == 0 {
		cX := fCenter.X + midU*fUAxis.X + midV*fVAxis.X
		cY := fCenter.Y + midU*fUAxis.Y + midV*fVAxis.Y
		cZ := fCenter.Z + midU*fUAxis.Z + midV*fVAxis.Z
		lenC := math.Sqrt(cX*cX + cY*cY + cZ*cZ)
		pos := s.GeodeticToECEF(math.Asin(cZ/lenC)*180/math.Pi, math.Atan2(cY, cX)*180/math.Pi, 0)
		return [3]float32{float32(pos.X), float32(pos.Y), float32(pos.Z)}, 10.0, [3]float32{float32(cX / lenC), float32(cY / lenC), float32(cZ / lenC)}
	}

	var sumX, sumY, sumZ float64
	for _, p := range splats {
		sumX += float64(p.Position[0])
		sumY += float64(p.Position[1])
		sumZ += float64(p.Position[2])
	}
	n := float64(len(splats))
	center := [3]float32{float32(sumX / n), float32(sumY / n), float32(sumZ / n)}

	var maxR float64
	for _, p := range splats {
		dx := float64(p.Position[0]) - float64(center[0])
		dy := float64(p.Position[1]) - float64(center[1])
		dz := float64(p.Position[2]) - float64(center[2])
		r := math.Sqrt(dx*dx + dy*dy + dz*dz)
		if r > maxR {
			maxR = r
		}
	}

	// Outward unit normal at patch center
	cX := fCenter.X + midU*fUAxis.X + midV*fVAxis.X
	cY := fCenter.Y + midU*fUAxis.Y + midV*fVAxis.Y
	cZ := fCenter.Z + midU*fUAxis.Z + midV*fVAxis.Z
	lenC := math.Sqrt(cX*cX + cY*cY + cZ*cZ)
	norm := [3]float32{float32(cX / lenC), float32(cY / lenC), float32(cZ / lenC)}

	return center, float32(maxR * 1.15), norm
}

// WriteContainer generates the full multi-LOD cubed-sphere terrain directly into a compressed indexed container.
func (tg *TerrainGenerator) WriteContainer(filePath string, patchesPerFaceAxis int) error {
	procCube := tg.NewProceduralCubedSphere(patchesPerFaceAxis)
	density := tg.EstimatePointDensity(0)

	header := splat.ContainerHeader{
		Radius:          tg.config.Spheroid.SemiMajorAxis,
		Flattening:      tg.config.Spheroid.Flattening,
		TotalSplatCount: uint64(density.TotalPoints),
		VirtualPoints:   uint64(density.TotalPoints),
		MeanSpacingKm:   density.MeanSpacingMeters / 1000.0,
		ChunkCount:      uint32(procCube.TotalChunks),
		MaxLODLevels:    5,
	}

	cw, err := splat.NewContainerWriter(filePath, header)
	if err != nil {
		return err
	}

	for chunkID := 0; chunkID < procCube.TotalChunks; chunkID++ {
		m := procCube.Manifests[chunkID]
		cm := splat.ChunkManifest{
			ChunkID: uint32(m.ID),
			Center:  m.Center,
			Radius:  m.Radius,
			Normal:  m.Normal,
			LatMin:  -90.0,
			LatMax:  90.0,
			LonMin:  -180.0,
			LonMax:  180.0,
			LODs:    make([]splat.ChunkLODEntry, 0, 5),
		}

		for lod := 0; lod < 5; lod++ {
			splats := procCube.SampleChunkLOD(chunkID, lod)
			entry, err := cw.WriteChunkLOD(chunkID, uint32(lod), splats)
			if err != nil {
				return err
			}
			cm.LODs = append(cm.LODs, *entry)
		}

		cw.AddChunkManifest(cm)
	}

	return cw.Finalize()
}
