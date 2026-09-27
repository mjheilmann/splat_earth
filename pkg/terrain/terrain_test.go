package terrain

import (
	"math"
	"testing"
)

func TestTerrainGenerator(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StepsLat = 20
	cfg.StepsLon = 30

	gen := NewTerrainGenerator(cfg)
	splats := gen.Generate()

	if len(splats) == 0 {
		t.Fatalf("Expected generated splats, got 0")
	}

	for i, s := range splats {
		// Verify position is finite
		for j := 0; j < 3; j++ {
			if math.IsNaN(float64(s.Position[j])) || math.IsInf(float64(s.Position[j]), 0) {
				t.Fatalf("Splat %d pos[%d] is invalid: %v", i, j, s.Position[j])
			}
		}

		// Verify scale > 0
		if s.Scale[0] <= 0 || s.Scale[1] <= 0 || s.Scale[2] <= 0 {
			t.Fatalf("Splat %d invalid scale: %v", i, s.Scale)
		}

		// Verify quaternion normalized
		lenQ := math.Sqrt(float64(s.Rotation[0]*s.Rotation[0] + s.Rotation[1]*s.Rotation[1] + s.Rotation[2]*s.Rotation[2] + s.Rotation[3]*s.Rotation[3]))
		if math.Abs(lenQ-1.0) > 0.05 {
			t.Fatalf("Splat %d quaternion not normalized: len=%v, q=%v", i, lenQ, s.Rotation)
		}

		// Verify alpha > 0
		if s.Color[3] == 0 {
			t.Fatalf("Splat %d alpha is 0", i)
		}
	}
}

func TestTerrainSlopeOrientation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ElevationScale = 20.0
	gen := NewTerrainGenerator(cfg)

	// Test slope gradient calculations at multiple points
	testCoords := [][2]float64{
		{0.0, 0.0},
		{45.0, 45.0},
		{-30.0, 120.0},
		{60.0, -90.0},
	}

	for _, coord := range testCoords {
		elev, slopeE, slopeN, _ := gen.sampleElevationAndSlopes(coord[0], coord[1])
		if math.IsNaN(elev) || math.IsNaN(slopeE) || math.IsNaN(slopeN) {
			t.Fatalf("NaN detected at lat=%v, lon=%v: elev=%v, slopeE=%v, slopeN=%v", coord[0], coord[1], elev, slopeE, slopeN)
		}

		// Calculate quaternion with and without slopes
		quatFlat := cfg.Spheroid.OrientationQuaternion(coord[0], coord[1], 0, 0)
		quatSloped := cfg.Spheroid.OrientationQuaternion(coord[0], coord[1], slopeE, slopeN)

		// Quaternions should be unit length
		lenFlat := math.Sqrt(quatFlat.W*quatFlat.W + quatFlat.X*quatFlat.X + quatFlat.Y*quatFlat.Y + quatFlat.Z*quatFlat.Z)
		lenSloped := math.Sqrt(quatSloped.W*quatSloped.W + quatSloped.X*quatSloped.X + quatSloped.Y*quatSloped.Y + quatSloped.Z*quatSloped.Z)

		if math.Abs(lenFlat-1.0) > 1e-4 || math.Abs(lenSloped-1.0) > 1e-4 {
			t.Fatalf("Quaternions not normalized: flat=%v, sloped=%v", lenFlat, lenSloped)
		}
	}
}

func TestFixedBudgetLODPopulation(t *testing.T) {
	cfg := DefaultConfig()
	// Simulate huge 500 Million point planetary grid
	cfg.StepsLat = 16000
	cfg.StepsLon = 32000
	cfg.AdaptiveDensity = false // Test exact uniform stride mapping

	gen := NewTerrainGenerator(cfg)
	proc := gen.NewProceduralCubedSphere(32) // 6144 chunks

	if proc.TotalChunks != 6144 {
		t.Fatalf("Expected 6144 chunks, got %d", proc.TotalChunks)
	}

	// Verify LOD 4 (Orbit) count per chunk is strictly bounded (target 12x12 ~ 13x13 splats, up to 2x for ocean chunks with sea surface)
	m0 := proc.Manifests[0]
	if m0.CountsLOD[4] > 400 {
		t.Fatalf("Expected Orbit LOD4 count <= 400 splats per chunk, got %d", m0.CountsLOD[4])
	}
	if m0.CountsLOD[3] > 1400 {
		t.Fatalf("Expected LOD3 count <= 1400 splats per chunk, got %d", m0.CountsLOD[3])
	}

	// Sample on demand for a chunk
	splatsLOD4 := proc.SampleChunkLOD(0, 4)
	if len(splatsLOD4) == 0 || len(splatsLOD4) > 400 {
		t.Fatalf("Expected LOD4 sample length in [1, 400], got %d", len(splatsLOD4))
	}
	if len(splatsLOD4) != m0.CountsLOD[4] {
		t.Fatalf("Expected sample length %d to match manifest count %d", len(splatsLOD4), m0.CountsLOD[4])
	}
}

func TestAdaptiveCurvatureSampling(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StepsLat = 4000
	cfg.StepsLon = 8000
	cfg.AdaptiveDensity = true
	cfg.CurvatureTolerance = 0.15

	gen := NewTerrainGenerator(cfg)
	proc := gen.NewProceduralCubedSphere(16) // 1536 chunks

	// Sample LOD 0 for multiple chunks
	// On ocean chunks (smooth), point count should be substantially lower than uniform grid
	uniformExpected := proc.PointsPerPatchSide * proc.PointsPerPatchSide
	splatsChunk0 := proc.SampleChunkLOD(0, 0)

	if len(splatsChunk0) == 0 {
		t.Fatalf("Expected generated adaptive splats, got 0")
	}

	// Verify all splats have positive scale and valid positions
	for i, s := range splatsChunk0 {
		if s.Scale[0] <= 0 || s.Scale[1] <= 0 || s.Scale[2] <= 0 {
			t.Fatalf("Invalid scale on splat %d: %v", i, s.Scale)
		}
		for j := 0; j < 3; j++ {
			if math.IsNaN(float64(s.Position[j])) || math.IsInf(float64(s.Position[j]), 0) {
				t.Fatalf("Invalid position on splat %d: %v", i, s.Position)
			}
		}
	}

	t.Logf("Uniform grid points/chunk: %d, Adaptive splats sampled: %d (Compression: %.1fx)",
		uniformExpected, len(splatsChunk0), float64(uniformExpected)/float64(len(splatsChunk0)))
}

func TestSteeredDensityFocusArea(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StepsLat = 100
	cfg.StepsLon = 200
	cfg.FocusArea = DensityZone{
		Enabled:           true,
		CenterLat:         0.0,
		CenterLon:         0.0,
		AreaSqMiles:       10.0,
		DensityMultiplier: 8.0,
	}

	gen := NewTerrainGenerator(cfg)

	// 1. Verify density multiplier evaluation
	multCenter := gen.GetDensityMultiplier(0.0, 0.0)
	if multCenter < 7.99 {
		t.Errorf("Expected center density multiplier ~8.0, got %v", multCenter)
	}

	multFar := gen.GetDensityMultiplier(60.0, 120.0)
	if math.Abs(multFar-1.0) > 1e-4 {
		t.Errorf("Expected far density multiplier 1.0, got %v", multFar)
	}

	// 2. Verify density metrics
	metrics := gen.EstimatePointDensity(0)
	if !metrics.FocusEnabled {
		t.Errorf("Expected FocusEnabled true in metrics")
	}
	if metrics.FocusSplatsPerKm2 < metrics.SplatsPerKm2*7.9 {
		t.Errorf("Expected FocusSplatsPerKm2 >= 8x base, got %v vs %v", metrics.FocusSplatsPerKm2, metrics.SplatsPerKm2)
	}
	if metrics.FocusMeanSpacingMeters > metrics.MeanSpacingMeters {
		t.Errorf("Expected FocusMeanSpacingMeters < base spacing, got %v vs %v", metrics.FocusMeanSpacingMeters, metrics.MeanSpacingMeters)
	}

	// 3. Generate procedural cubed sphere
	proc := gen.NewProceduralCubedSphere(4) // 96 chunks
	// Chunk 5 is on Face 0 covering center (0 deg Lat, 0 deg Lon)
	splatsFocus := proc.SampleChunkLOD(5, 0)
	if len(splatsFocus) == 0 {
		t.Fatalf("Expected non-empty focus splats")
	}

	// Chunk on opposite side (e.g. Face 2: -X 180 deg Lon)
	oppositeChunkID := 2 * 16 // Face 2
	splatsOpposite := proc.SampleChunkLOD(oppositeChunkID, 0)

	t.Logf("Focus chunk splats: %d, Opposite chunk splats: %d (Density ratio: %.2fx)",
		len(splatsFocus), len(splatsOpposite), float64(len(splatsFocus))/float64(len(splatsOpposite)))

	if len(splatsFocus) <= len(splatsOpposite) {
		t.Errorf("Expected focus chunk to produce more splats than opposite chunk, got %d vs %d",
			len(splatsFocus), len(splatsOpposite))
	}

	// Verify all splats are valid
	for i, s := range splatsFocus {
		if s.Scale[0] <= 0 || s.Scale[1] <= 0 || s.Scale[2] <= 0 {
			t.Fatalf("Focus splat %d has non-positive scale: %v", i, s.Scale)
		}
		for j := 0; j < 3; j++ {
			if math.IsNaN(float64(s.Position[j])) || math.IsInf(float64(s.Position[j]), 0) {
				t.Fatalf("Focus splat %d has invalid pos[%d]: %v", i, j, s.Position[j])
			}
		}
	}
}

func TestSeparatedPlanetAndFocusDensity(t *testing.T) {
	cfg := DefaultConfig()
	// Low density planet base (~2.8 km spacing)
	cfg.StepsLat = 80
	cfg.StepsLon = 160
	cfg.FocusArea = DensityZone{
		Enabled:           true,
		CenterLat:         10.0,
		CenterLon:         20.0,
		AreaSqMiles:       10.0, // ~25.9 km2, radius ~2.87 km
		DensityMultiplier: 64.0, // 64x boost (~70m spacing)
	}

	gen := NewTerrainGenerator(cfg)

	// Sample face containing the focus area
	// Find chunk that contains (Lat 10, Lon 20)
	splatsWithFocus := gen.GenerateCubedSphereSplats(4)
	if len(splatsWithFocus) == 0 {
		t.Fatalf("Expected non-empty splats")
	}

	// Verify that splats near the focus center have significantly smaller scales than splats far away
	var focusMinScale float32 = 1e9
	var farMaxScale float32 = 0.0

	for _, s := range splatsWithFocus {
		px := float64(s.Position[0])
		py := float64(s.Position[1])
		pz := float64(s.Position[2])
		lenP := math.Sqrt(px*px + py*py + pz*pz)
		if lenP < 1e-4 {
			continue
		}
		latDeg := math.Asin(math.Max(-1.0, math.Min(1.0, pz/lenP))) * 180.0 / math.Pi
		lonDeg := math.Atan2(py, px) * 180.0 / math.Pi

		dist := cfg.Spheroid.SurfaceDistance(latDeg, lonDeg, 10.0, 20.0)

		if dist < 2.5 { // Inside the 2.87 km focus radius
			if s.Scale[0] < focusMinScale {
				focusMinScale = s.Scale[0]
			}
		} else if dist > 40.0 { // Far outside on the planet
			if s.Scale[0] > farMaxScale {
				farMaxScale = s.Scale[0]
			}
		}
	}

	t.Logf("Focus min splat scale: %f, Far max splat scale: %f (Resolution scale ratio: %.1fx)",
		focusMinScale, farMaxScale, float64(farMaxScale)/float64(focusMinScale))

	if focusMinScale >= farMaxScale {
		t.Errorf("Expected focus area splats to have smaller scale than far planet splats, got focus=%f, far=%f",
			focusMinScale, farMaxScale)
	}
	if farMaxScale/focusMinScale < 4.0 {
		t.Errorf("Expected at least 4x scale difference between focus and planet base, got %.1fx",
			float64(farMaxScale)/float64(focusMinScale))
	}
}

func TestMultiTierDEMSampling(t *testing.T) {
	cfg := DefaultConfig()
	gen := NewTerrainGenerator(cfg)

	// 1. Inside Seattle GeoTIFF area: Lat 47.6205 (Space Needle), Lon -122.3493
	hSeattle := gen.sampleHeight(47.6205, -122.3493)
	if hSeattle < 0.55 || hSeattle > 0.65 {
		t.Errorf("Expected normalized elevation near sea level ~0.56 for Seattle, got %f", hSeattle)
	}

	// 2. Global DEM Everest: Lat 27.98, Lon 86.92
	hEverest := gen.sampleHeight(27.98, 86.92)
	if hEverest < 0.90 {
		t.Errorf("Expected Everest normalized elevation > 0.90, got %f", hEverest)
	}
}


