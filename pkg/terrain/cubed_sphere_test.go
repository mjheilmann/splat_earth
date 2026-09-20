package terrain

import (
	"math"
	"testing"
)

func TestCubedSphereTerrain(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StepsLat = 60
	cfg.StepsLon = 120
	gen := NewTerrainGenerator(cfg)

	// Generate 6 faces x 2x2 = 24 patches
	patchesPerAxis := 2
	terrain := gen.GenerateCubedSphereTerrain(patchesPerAxis)

	expectedChunks := 6 * patchesPerAxis * patchesPerAxis
	if len(terrain.Chunks) != expectedChunks {
		t.Fatalf("expected %d chunks, got %d", expectedChunks, len(terrain.Chunks))
	}
	if len(terrain.Manifests) != expectedChunks {
		t.Fatalf("expected %d manifests, got %d", expectedChunks, len(terrain.Manifests))
	}

	for i, ch := range terrain.Chunks {
		if ch.ID != i {
			t.Errorf("chunk ID mismatch: expected %d, got %d", i, ch.ID)
		}
		if len(ch.LODs[0]) == 0 {
			t.Errorf("chunk %d has 0 splats at LOD 0", ch.ID)
		}
		if len(ch.LODs[1]) >= len(ch.LODs[0]) {
			t.Errorf("chunk %d LOD 1 count (%d) should be less than LOD 0 count (%d)",
				ch.ID, len(ch.LODs[1]), len(ch.LODs[0]))
		}
		if len(ch.LODs[2]) >= len(ch.LODs[1]) {
			t.Errorf("chunk %d LOD 2 count (%d) should be less than LOD 1 count (%d)",
				ch.ID, len(ch.LODs[2]), len(ch.LODs[1]))
		}
		if ch.Radius <= 0 {
			t.Errorf("chunk %d has invalid radius %f", ch.ID, ch.Radius)
		}

		// Verify normal is unit vector
		normLen := math.Hypot(float64(ch.Normal[0]), math.Hypot(float64(ch.Normal[1]), float64(ch.Normal[2])))
		if math.Abs(normLen-1.0) > 0.05 {
			t.Errorf("chunk %d normal is not unit length: %f", ch.ID, normLen)
		}

		// Verify splats are finite numbers
		for _, s := range ch.LODs[0] {
			for c := 0; c < 3; c++ {
				if math.IsNaN(float64(s.Position[c])) || math.IsInf(float64(s.Position[c]), 0) {
					t.Fatalf("chunk %d has NaN/Inf position: %v", ch.ID, s.Position)
				}
				if s.Scale[c] <= 0 {
					t.Fatalf("chunk %d has non-positive scale: %v", ch.ID, s.Scale)
				}
			}
		}
	}

	// Test flat splats generator helper
	allSplats := gen.GenerateCubedSphereSplats(patchesPerAxis)
	if len(allSplats) == 0 {
		t.Fatalf("GenerateCubedSphereSplats returned 0 splats")
	}
}

func TestFaceBases(t *testing.T) {
	for f := 0; f < 6; f++ {
		face := CubeFace(f)
		c, u, v := face.FaceBasis()

		// Center should be cardinal axis unit vector
		lenC := math.Sqrt(c.X*c.X + c.Y*c.Y + c.Z*c.Z)
		if math.Abs(lenC-1.0) > 1e-6 {
			t.Errorf("face %d center length != 1: %f", f, lenC)
		}

		// u and v should be orthogonal to c and to each other
		dotUC := u.X*c.X + u.Y*c.Y + u.Z*c.Z
		dotVC := v.X*c.X + v.Y*c.Y + v.Z*c.Z
		dotUV := u.X*v.X + u.Y*v.Y + u.Z*v.Z

		if math.Abs(dotUC) > 1e-6 {
			t.Errorf("face %d u not orthogonal to c: %f", f, dotUC)
		}
		if math.Abs(dotVC) > 1e-6 {
			t.Errorf("face %d v not orthogonal to c: %f", f, dotVC)
		}
		if math.Abs(dotUV) > 1e-6 {
			t.Errorf("face %d u not orthogonal to v: %f", f, dotUV)
		}
	}
}

func TestEstimatePointDensity(t *testing.T) {
	cfg := DefaultConfig()
	gen := NewTerrainGenerator(cfg)

	// Test density estimation for different scales
	counts := []int{20000, 100000, 500000, 2000000, 5000000}
	for _, n := range counts {
		metrics := gen.EstimatePointDensity(n)
		if metrics.TotalPoints != n {
			t.Errorf("expected totalPoints %d, got %d", n, metrics.TotalPoints)
		}
		if metrics.SurfaceAreaKm2 <= 0 {
			t.Errorf("invalid surface area %f", metrics.SurfaceAreaKm2)
		}
		if metrics.SplatsPerKm2 <= 0 {
			t.Errorf("invalid splats per km2 %f", metrics.SplatsPerKm2)
		}
		if metrics.MeanSpacingMeters <= 0 {
			t.Errorf("invalid mean spacing %f", metrics.MeanSpacingMeters)
		}
		if metrics.RecommendedPatchesPerFace < 2 || metrics.RecommendedPatchesPerFace > 16 {
			t.Errorf("recommended patches out of range: %d", metrics.RecommendedPatchesPerFace)
		}
	}
}

func TestAutoAdaptiveChunking(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StepsLat = 80
	cfg.StepsLon = 160
	gen := NewTerrainGenerator(cfg)

	// Passing 0 triggers auto-adaptive chunk selection
	chunked := gen.GenerateCubedSphereTerrain(0)
	if len(chunked.Chunks) < 6 {
		t.Fatalf("expected at least 6 chunks, got %d", len(chunked.Chunks))
	}
	if len(chunked.Manifests) != len(chunked.Chunks) {
		t.Fatalf("chunk and manifest length mismatch")
	}
}
