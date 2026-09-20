package splat

// Splat represents a single 3D Gaussian Splat.
type Splat struct {
	// Position in 3D world space (X, Y, Z)
	Position [3]float32

	// Scale along local principal axes (Sx, Sy, Sz)
	// For terrain: Sx and Sy represent surface tangent extent, Sz represents height thickness
	Scale [3]float32

	// Orientation unit quaternion (W, X, Y, Z)
	Rotation [4]float32

	// Color in RGBA [0..255]
	Color [4]uint8
}

// SplatSet contains a collection of 3D Gaussian Splats with bounding info.
type SplatSet struct {
	Splats []Splat
	Min    [3]float32
	Max    [3]float32
}

// NewSplatSet initializes a SplatSet from a slice of splats and calculates bounds.
func NewSplatSet(splats []Splat) *SplatSet {
	set := &SplatSet{
		Splats: splats,
		Min:    [3]float32{1e30, 1e30, 1e30},
		Max:    [3]float32{-1e30, -1e30, -1e30},
	}
	for _, s := range splats {
		for i := 0; i < 3; i++ {
			if s.Position[i] < set.Min[i] {
				set.Min[i] = s.Position[i]
			}
			if s.Position[i] > set.Max[i] {
				set.Max[i] = s.Position[i]
			}
		}
	}
	return set
}
