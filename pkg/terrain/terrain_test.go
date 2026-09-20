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
		elev, slopeE, slopeN := gen.sampleElevationAndSlopes(coord[0], coord[1])
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

	// Verify LOD 4 (Orbit) count per chunk is strictly bounded (target 12x12 ~ 13x13 splats)
	m0 := proc.Manifests[0]
	if m0.CountsLOD[4] > 200 {
		t.Fatalf("Expected Orbit LOD4 count <= 200 splats per chunk, got %d", m0.CountsLOD[4])
	}
	if m0.CountsLOD[3] > 700 {
		t.Fatalf("Expected LOD3 count <= 700 splats per chunk, got %d", m0.CountsLOD[3])
	}

	// Sample on demand for a chunk
	splatsLOD4 := proc.SampleChunkLOD(0, 4)
	if len(splatsLOD4) == 0 || len(splatsLOD4) > 200 {
		t.Fatalf("Expected LOD4 sample length in [1, 200], got %d", len(splatsLOD4))
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
