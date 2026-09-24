package terrain

import (
	"math"
	"testing"
)

func TestUTM10NRoundTrip(t *testing.T) {
	// Seattle Space Needle
	lat := 47.6205
	lon := -122.3493

	easting, northing := LatLonToUTM10N(lat, lon)
	latOut, lonOut := UTM10NToLatLon(easting, northing)

	if math.Abs(lat-latOut) > 1e-5 {
		t.Errorf("Latitude mismatch: %f vs %f", lat, latOut)
	}
	if math.Abs(lon-lonOut) > 1e-5 {
		t.Errorf("Longitude mismatch: %f vs %f", lon, lonOut)
	}
}

func TestGeoTIFFTileAndMosaicSampling(t *testing.T) {
	tifs := FindDefaultGeoTIFFs()
	if len(tifs) == 0 {
		t.Skip("No GeoTIFF LiDAR files found, skipping")
	}

	mosaic, err := LoadGeoTIFFMosaic(tifs)
	if err != nil {
		t.Fatalf("Failed loading GeoTIFF mosaic: %v", err)
	}

	if len(mosaic.Tiles) < 4 {
		t.Errorf("Expected at least 4 tiles, got %d", len(mosaic.Tiles))
	}

	// Sample Queen Anne / Seattle Hill (Lat 47.63, Lon -122.35)
	elev, ok := mosaic.Sample(47.63, -122.35)
	if !ok {
		t.Fatalf("Expected Queen Anne within mosaic bounds, got not ok")
	}
	t.Logf("Sampled Seattle elevation: %.2f meters", elev)
	if elev < 0 || elev > 250 {
		t.Errorf("Expected reasonable Seattle elevation [0, 250]m, got %f", elev)
	}

	// Sample far outside (e.g. Hawaii / Pacific) -> should return ok=false
	_, okFar := mosaic.Sample(20.0, -155.0)
	if okFar {
		t.Errorf("Expected ok=false for Hawaii coordinate in King County mosaic")
	}
}
