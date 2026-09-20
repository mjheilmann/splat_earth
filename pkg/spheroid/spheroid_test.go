package spheroid

import (
	"math"
	"testing"
)

func TestGeodeticECEFRoundTrip(t *testing.T) {
	s := WGS84()

	testCases := []struct {
		name string
		lat  float64
		lon  float64
		h    float64
	}{
		{"Equator Origin", 0.0, 0.0, 0.0},
		{"North Pole", 90.0, 0.0, 500.0},
		{"South Pole", -90.0, 45.0, 1000.0},
		{"Mid Latitude High Elev", 45.0, 90.0, 8848.0},
		{"Southern Hemisphere", -33.8688, 151.2093, 50.0}, // Sydney
		{"Western Hemisphere", 37.7749, -122.4194, 100.0}, // SF
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ecef := s.GeodeticToECEF(tc.lat, tc.lon, tc.h)
			lat, lon, h := s.ECEFToGeodetic(ecef)

			if math.Abs(lat-tc.lat) > 1e-5 {
				t.Errorf("Lat mismatch: got %v, want %v", lat, tc.lat)
			}
			if math.Abs(tc.lat) < 89.9 { // Lon is degenerate at poles
				lonDiff := math.Abs(lon - tc.lon)
				if lonDiff > 180.0 {
					lonDiff = math.Abs(lonDiff - 360.0)
				}
				if lonDiff > 1e-5 {
					t.Errorf("Lon mismatch: got %v, want %v", lon, tc.lon)
				}
			}
			if math.Abs(h-tc.h) > 1e-2 {
				t.Errorf("Height mismatch: got %v, want %v", h, tc.h)
			}
		})
	}
}

func TestTangentBasisOrthonormal(t *testing.T) {
	s := WGS84()
	lats := []float64{-60, -30, 0, 30, 60}
	lons := []float64{-120, -45, 0, 45, 120}

	for _, lat := range lats {
		for _, lon := range lons {
			east, north, up := s.TangentBasis(lat, lon)

			// Check unit lengths
			lenEast := math.Sqrt(east.X*east.X + east.Y*east.Y + east.Z*east.Z)
			lenNorth := math.Sqrt(north.X*north.X + north.Y*north.Y + north.Z*north.Z)
			lenUp := math.Sqrt(up.X*up.X + up.Y*up.Y + up.Z*up.Z)

			if math.Abs(lenEast-1.0) > 1e-6 || math.Abs(lenNorth-1.0) > 1e-6 || math.Abs(lenUp-1.0) > 1e-6 {
				t.Errorf("Basis vector not unit length at lat=%v, lon=%v", lat, lon)
			}

			// Check orthogonality
			dotEN := east.X*north.X + east.Y*north.Y + east.Z*north.Z
			dotEU := east.X*up.X + east.Y*up.Y + east.Z*up.Z
			dotNU := north.X*up.X + north.Y*up.Y + north.Z*up.Z

			if math.Abs(dotEN) > 1e-6 || math.Abs(dotEU) > 1e-6 || math.Abs(dotNU) > 1e-6 {
				t.Errorf("Basis vectors not orthogonal at lat=%v, lon=%v: EN=%v, EU=%v, NU=%v", lat, lon, dotEN, dotEU, dotNU)
			}
		}
	}
}

func TestOrientationQuaternion(t *testing.T) {
	s := WGS84()
	q := s.OrientationQuaternion(45.0, 30.0, 0.1, -0.05)
	lenQ := math.Sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if math.Abs(lenQ-1.0) > 1e-6 {
		t.Errorf("Quaternion not normalized: %v (length %v)", q, lenQ)
	}
}
