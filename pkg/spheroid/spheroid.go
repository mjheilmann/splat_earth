package spheroid

import (
	"math"
)

// Spheroid defines an oblate spheroid (ellipsoid of revolution).
// SemiMajorAxis (a) is the equatorial radius.
// SemiMinorAxis (b) is the polar radius.
// Flattening f = (a - b) / a
// Eccentricity squared e^2 = (a^2 - b^2) / a^2 = 2f - f^2
type Spheroid struct {
	SemiMajorAxis float64 // a (meters or custom units)
	SemiMinorAxis float64 // b (meters or custom units)
	Flattening    float64 // f
	Eccentricity2 float64 // e^2
}

// WGS84 returns the standard WGS-84 reference ellipsoid in real-world meters.
func WGS84() Spheroid {
	a := 6378137.0
	f := 1.0 / 298.257223563
	b := a * (1.0 - f)
	e2 := (a*a - b*b) / (a * a)
	return Spheroid{
		SemiMajorAxis: a,
		SemiMinorAxis: b,
		Flattening:    f,
		Eccentricity2: e2,
	}
}

// NewSpheroid creates an oblate spheroid scaled for model/visualization units with custom flattening.
func NewSpheroid(equatorialRadius float64, flattening float64) Spheroid {
	b := equatorialRadius * (1.0 - flattening)
	e2 := (equatorialRadius*equatorialRadius - b*b) / (equatorialRadius * equatorialRadius)
	return Spheroid{
		SemiMajorAxis: equatorialRadius,
		SemiMinorAxis: b,
		Flattening:    flattening,
		Eccentricity2: e2,
	}
}

// Vec3 represents a 3D vector.
type Vec3 struct {
	X, Y, Z float64
}

// Quat represents a unit quaternion (W, X, Y, Z) for 3D rotations.
type Quat struct {
	W, X, Y, Z float64
}

// GeodeticToECEF converts geodetic coordinates (lat, lon in degrees, height above ellipsoid)
// to Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates (X, Y, Z).
func (s Spheroid) GeodeticToECEF(latDeg, lonDeg, height float64) Vec3 {
	latRad := latDeg * math.Pi / 180.0
	lonRad := lonDeg * math.Pi / 180.0

	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	sinLon := math.Sin(lonRad)
	cosLon := math.Cos(lonRad)

	// Radius of curvature in the prime vertical: N(phi) = a / sqrt(1 - e^2 * sin^2(phi))
	N := s.SemiMajorAxis / math.Sqrt(1.0-s.Eccentricity2*sinLat*sinLat)

	x := (N + height) * cosLat * cosLon
	y := (N + height) * cosLat * sinLon
	z := (N*(1.0-s.Eccentricity2) + height) * sinLat

	return Vec3{X: x, Y: y, Z: z}
}

// ECEFToGeodetic converts ECEF Cartesian coordinates (X, Y, Z) to geodetic coordinates (latDeg, lonDeg, height).
// Uses Bowring's iterative method for high precision on oblate spheroids.
func (s Spheroid) ECEFToGeodetic(pos Vec3) (latDeg, lonDeg, height float64) {
	lonRad := math.Atan2(pos.Y, pos.X)
	p := math.Hypot(pos.X, pos.Y)

	if p < 1e-6 {
		// Near poles
		if pos.Z > 0 {
			return 90.0, 0.0, pos.Z - s.SemiMinorAxis
		}
		return -90.0, 0.0, -pos.Z - s.SemiMinorAxis
	}

	// Bowring's approximation
	ePrime2 := (s.SemiMajorAxis*s.SemiMajorAxis - s.SemiMinorAxis*s.SemiMinorAxis) / (s.SemiMinorAxis * s.SemiMinorAxis)
	theta := math.Atan2(pos.Z*s.SemiMajorAxis, p*s.SemiMinorAxis)

	sinTheta := math.Sin(theta)
	cosTheta := math.Cos(theta)

	latRad := math.Atan2(
		pos.Z+ePrime2*s.SemiMinorAxis*sinTheta*sinTheta*sinTheta,
		p-s.Eccentricity2*s.SemiMajorAxis*cosTheta*cosTheta*cosTheta,
	)

	sinLat := math.Sin(latRad)
	N := s.SemiMajorAxis / math.Sqrt(1.0-s.Eccentricity2*sinLat*sinLat)
	h := p/math.Cos(latRad) - N

	return latRad * 180.0 / math.Pi, lonRad * 180.0 / math.Pi, h
}

// SurfaceNormal computes the outward unit surface normal on the ellipsoid at the given geodetic coordinates.
func (s Spheroid) SurfaceNormal(latDeg, lonDeg float64) Vec3 {
	latRad := latDeg * math.Pi / 180.0
	lonRad := lonDeg * math.Pi / 180.0

	return Vec3{
		X: math.Cos(latRad) * math.Cos(lonRad),
		Y: math.Cos(latRad) * math.Sin(lonRad),
		Z: math.Sin(latRad),
	}
}

// TangentBasis returns the orthonormal local East, North, and Up (Normal) unit vectors at (latDeg, lonDeg).
// East  = tangent along longitude (parallel to equator)
// North = tangent along latitude (towards north pole)
// Up    = outward normal perpendicular to ellipsoid surface
func (s Spheroid) TangentBasis(latDeg, lonDeg float64) (east, north, up Vec3) {
	latRad := latDeg * math.Pi / 180.0
	lonRad := lonDeg * math.Pi / 180.0

	sinLat, cosLat := math.Sin(latRad), math.Cos(latRad)
	sinLon, cosLon := math.Sin(lonRad), math.Cos(lonRad)

	east = Vec3{
		X: -sinLon,
		Y: cosLon,
		Z: 0,
	}

	north = Vec3{
		X: -sinLat * cosLon,
		Y: -sinLat * sinLon,
		Z: cosLat,
	}

	up = Vec3{
		X: cosLat * cosLon,
		Y: cosLat * sinLon,
		Z: sinLat,
	}

	return east, north, up
}

// OrientationQuaternion computes the rotation quaternion that transforms a canonical local splat
// (where X = East, Y = North, Z = Up) into the world ECEF coordinate frame at (latDeg, lonDeg).
// Incorporates physical terrain slope gradients (slopeEast = dh/dx, slopeNorth = dh/dy) to match the true terrain normal.
func (s Spheroid) OrientationQuaternion(latDeg, lonDeg float64, slopeEast, slopeNorth float64) Quat {
	east, north, up := s.TangentBasis(latDeg, lonDeg)

	if slopeEast != 0 || slopeNorth != 0 {
		// True terrain surface normal: Up - slopeEast*East - slopeNorth*North
		terrainNormal := Vec3{
			X: up.X - slopeEast*east.X - slopeNorth*north.X,
			Y: up.Y - slopeEast*east.Y - slopeNorth*north.Y,
			Z: up.Z - slopeEast*east.Z - slopeNorth*north.Z,
		}
		lenNorm := math.Sqrt(terrainNormal.X*terrainNormal.X + terrainNormal.Y*terrainNormal.Y + terrainNormal.Z*terrainNormal.Z)
		if lenNorm > 1e-9 {
			up = Vec3{X: terrainNormal.X / lenNorm, Y: terrainNormal.Y / lenNorm, Z: terrainNormal.Z / lenNorm}

			// Project East onto the terrain tangent plane: east_proj = east - (east . up)*up
			dotE := east.X*up.X + east.Y*up.Y + east.Z*up.Z
			eastProj := Vec3{X: east.X - dotE*up.X, Y: east.Y - dotE*up.Y, Z: east.Z - dotE*up.Z}
			lenE := math.Sqrt(eastProj.X*eastProj.X + eastProj.Y*eastProj.Y + eastProj.Z*eastProj.Z)
			if lenE > 1e-9 {
				east = Vec3{X: eastProj.X / lenE, Y: eastProj.Y / lenE, Z: eastProj.Z / lenE}
			} else {
				east = cross(Vec3{X: 0, Y: 0, Z: 1}, up)
				lenE2 := math.Sqrt(east.X*east.X + east.Y*east.Y + east.Z*east.Z)
				if lenE2 > 1e-9 {
					east = Vec3{X: east.X / lenE2, Y: east.Y / lenE2, Z: east.Z / lenE2}
				}
			}
			north = cross(up, east)
			lenN := math.Sqrt(north.X*north.X + north.Y*north.Y + north.Z*north.Z)
			if lenN > 1e-9 {
				north = Vec3{X: north.X / lenN, Y: north.Y / lenN, Z: north.Z / lenN}
			}
		}
	}

	// Matrix columns are east, north, up:
	// M = [ east.X  north.X  up.X ]
	//     [ east.Y  north.Y  up.Y ]
	//     [ east.Z  north.Z  up.Z ]
	return matrixToQuaternion(
		east.X, north.X, up.X,
		east.Y, north.Y, up.Y,
		east.Z, north.Z, up.Z,
	)
}

func cross(a, b Vec3) Vec3 {
	return Vec3{
		X: a.Y*b.Z - a.Z*b.Y,
		Y: a.Z*b.X - a.X*b.Z,
		Z: a.X*b.Y - a.Y*b.X,
	}
}

// matrixToQuaternion converts a 3x3 rotation matrix to a normalized quaternion (W, X, Y, Z).
func matrixToQuaternion(m00, m01, m02, m10, m11, m12, m20, m21, m22 float64) Quat {
	tr := m00 + m11 + m22
	var q Quat

	if tr > 0 {
		S := math.Sqrt(tr+1.0) * 2
		q.W = 0.25 * S
		q.X = (m21 - m12) / S
		q.Y = (m02 - m20) / S
		q.Z = (m10 - m01) / S
	} else if (m00 > m11) && (m00 > m22) {
		S := math.Sqrt(1.0+m00-m11-m22) * 2
		q.W = (m21 - m12) / S
		q.X = 0.25 * S
		q.Y = (m01 + m10) / S
		q.Z = (m02 + m20) / S
	} else if m11 > m22 {
		S := math.Sqrt(1.0+m11-m00-m22) * 2
		q.W = (m02 - m20) / S
		q.X = (m01 + m10) / S
		q.Y = 0.25 * S
		q.Z = (m12 + m21) / S
	} else {
		S := math.Sqrt(1.0+m22-m00-m11) * 2
		q.W = (m10 - m01) / S
		q.X = (m02 + m20) / S
		q.Y = (m12 + m21) / S
		q.Z = 0.25 * S
	}

	// Normalize
	lenQ := math.Sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if lenQ > 1e-9 {
		q.W /= lenQ
		q.X /= lenQ
		q.Y /= lenQ
		q.Z /= lenQ
	} else {
		q = Quat{W: 1, X: 0, Y: 0, Z: 0}
	}

	return q
}
