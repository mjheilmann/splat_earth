package terrain

import (
	"math"
	"testing"
)

func TestElevationRasterSampling(t *testing.T) {
	rasterPath := FindDefaultElevationRaster()
	if rasterPath == "" {
		t.Skip("Elevation map image not present, skipping real raster test")
	}

	r, err := LoadElevationRaster(rasterPath)
	if err != nil {
		t.Fatalf("Failed to load elevation raster: %v", err)
	}

	if r.Width != 21600 || r.Height != 10800 {
		t.Errorf("Expected 21600x10800, got %dx%d", r.Width, r.Height)
	}

	// 1. Everest (Lat 27.98, Lon 86.92) - should be high elevation (~1.0)
	everest := r.Sample(27.98, 86.92)
	if everest < 0.90 {
		t.Errorf("Expected Everest elevation > 0.90, got %f", everest)
	}

	// 2. Mariana Trench (Lat 11.35, Lon 142.2) - should be deep ocean trench (< 0.15)
	trench := r.Sample(11.35, 142.2)
	if trench > 0.15 {
		t.Errorf("Expected Mariana Trench elevation < 0.15, got %f", trench)
	}

	// 3. Amazon Basin (Lat -3.0, Lon -60.0) - near sea level (~0.56 - 0.60)
	amazon := r.Sample(-3.0, -60.0)
	if amazon < 0.50 || amazon > 0.65 {
		t.Errorf("Expected Amazon Basin near sea level ~0.56, got %f", amazon)
	}

	// 4. Interpolation smoothness across 360 wrap
	wrapA := r.Sample(0.0, 179.99)
	wrapB := r.Sample(0.0, -179.99)
	if math.Abs(wrapA-wrapB) > 0.05 {
		t.Errorf("Discontinuity across meridian wrap: %f vs %f", wrapA, wrapB)
	}
}
