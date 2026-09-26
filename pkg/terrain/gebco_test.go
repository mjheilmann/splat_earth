package terrain

import (
	"testing"
)

func TestGEBCOMosaicSampling(t *testing.T) {
	files := FindDefaultGEBCOFiles()
	if len(files) < 8 {
		t.Skip("GEBCO files not found in standard directories, skipping integration test")
	}

	mosaic, err := LoadGEBCOMosaic(files)
	if err != nil {
		t.Fatalf("failed to load GEBCO mosaic: %v", err)
	}

	if mosaic.GlobalWidth != 43200 || mosaic.GlobalHeight != 21600 {
		t.Errorf("expected 43200x21600 dimensions, got %dx%d", mosaic.GlobalWidth, mosaic.GlobalHeight)
	}

	seaLevel := 0.56215

	// 1. Sample Orlando / Central Florida (Lat 28.5, Lon -81.3)
	elevOrlando, isLand := mosaic.Sample(seaLevel, 28.5, -81.3)
	if !isLand || elevOrlando < seaLevel {
		t.Errorf("expected Orlando to be land with elevation >= seaLevel, got elev=%f, isLand=%v", elevOrlando, isLand)
	}

	// 2. Sample Ocean / Deep Pacific (Lat 11.35, Lon 142.2)
	elevOcean, isLandOcean := mosaic.Sample(seaLevel, 11.35, 142.2)
	if isLandOcean || elevOcean >= seaLevel {
		t.Errorf("expected ocean to have isLand=false and elev < seaLevel, got elev=%f, isLand=%v", elevOcean, isLandOcean)
	}

	// 3. Sample Mt. Everest (Lat 27.9881, Lon 86.9250)
	elevEverest, isLandEverest := mosaic.Sample(seaLevel, 27.9881, 86.9250)
	if !isLandEverest || elevEverest < 0.90 {
		t.Errorf("expected Everest to be land with high normalized elevation, got elev=%f, isLand=%v", elevEverest, isLandEverest)
	}

	// 4. Test quadrant boundary continuity (e.g. crossing Lon 0.0 or Equator Lat 0.0)
	eNorth, _ := mosaic.Sample(seaLevel, 0.01, 0.0)
	eSouth, _ := mosaic.Sample(seaLevel, -0.01, 0.0)
	if diff := mathAbs(eNorth - eSouth); diff > 0.05 {
		t.Errorf("discontinuity across Equator: eNorth=%f, eSouth=%f (diff=%f)", eNorth, eSouth, diff)
	}

	eWest, _ := mosaic.Sample(seaLevel, 45.0, -0.01)
	eEast, _ := mosaic.Sample(seaLevel, 45.0, 0.01)
	if diff := mathAbs(eWest - eEast); diff > 0.05 {
		t.Errorf("discontinuity across Prime Meridian: eWest=%f, eEast=%f (diff=%f)", eWest, eEast, diff)
	}
}

func mathAbs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
