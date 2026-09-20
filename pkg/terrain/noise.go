package terrain

import (
	"math"
)

// PerlinNoise implements 3D gradient noise for spherical and planetary terrain generation.
type PerlinNoise struct {
	p [512]int
}

// NewPerlinNoise creates a seeded Perlin noise generator.
func NewPerlinNoise(seed int64) *PerlinNoise {
	pn := &PerlinNoise{}
	permutation := [256]int{
		151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142,
		8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117,
		35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
		134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41,
		55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89,
		18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226,
		250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182,
		189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43,
		172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97,
		228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239,
		107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
		138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
	}

	// Pseudo-randomize with seed
	for i := 0; i < 256; i++ {
		j := (int(seed*31) + i*7 + 13) % 256
		if j < 0 {
			j += 256
		}
		permutation[i], permutation[j] = permutation[j], permutation[i]
	}

	for i := 0; i < 256; i++ {
		pn.p[i] = permutation[i]
		pn.p[256+i] = permutation[i]
	}

	return pn
}

// Noise3D computes 3D Perlin noise at (x, y, z), returning values in approx [-1, 1].
func (pn *PerlinNoise) Noise3D(x, y, z float64) float64 {
	X := int(math.Floor(x)) & 255
	Y := int(math.Floor(y)) & 255
	Z := int(math.Floor(z)) & 255

	x -= math.Floor(x)
	y -= math.Floor(y)
	z -= math.Floor(z)

	u := fade(x)
	v := fade(y)
	w := fade(z)

	A := pn.p[X] + Y
	AA := pn.p[A] + Z
	AB := pn.p[A+1] + Z
	B := pn.p[X+1] + Y
	BA := pn.p[B] + Z
	BB := pn.p[B+1] + Z

	return lerp(w,
		lerp(v,
			lerp(u, grad(pn.p[AA], x, y, z), grad(pn.p[BA], x-1, y, z)),
			lerp(u, grad(pn.p[AB], x, y-1, z), grad(pn.p[BB], x-1, y-1, z))),
		lerp(v,
			lerp(u, grad(pn.p[AA+1], x, y, z-1), grad(pn.p[BA+1], x-1, y, z-1)),
			lerp(u, grad(pn.p[AB+1], x, y-1, z-1), grad(pn.p[BB+1], x-1, y-1, z-1))))
}

// FBM computes Fractal Brownian Motion (octaves of noise) at (x, y, z).
func (pn *PerlinNoise) FBM(x, y, z float64, octaves int, persistence, lacunarity float64) float64 {
	total := 0.0
	frequency := 1.0
	amplitude := 1.0
	maxValue := 0.0

	for i := 0; i < octaves; i++ {
		total += pn.Noise3D(x*frequency, y*frequency, z*frequency) * amplitude
		maxValue += amplitude
		amplitude *= persistence
		frequency *= lacunarity
	}

	return total / maxValue
}

// RidgeNoise creates sharp mountain ridges.
func (pn *PerlinNoise) RidgeNoise(x, y, z float64, octaves int, persistence, lacunarity float64) float64 {
	total := 0.0
	frequency := 1.0
	amplitude := 1.0
	maxValue := 0.0

	for i := 0; i < octaves; i++ {
		n := math.Abs(pn.Noise3D(x*frequency, y*frequency, z*frequency))
		n = 1.0 - n // Invert so peaks are sharp
		n = n * n   // Square for sharp ridges
		total += n * amplitude
		maxValue += amplitude
		amplitude *= persistence
		frequency *= lacunarity
	}

	return total / maxValue
}

// AlpineRidgeNoise creates knife-edge alpine peaks, arêtes, and massifs with power exponent shaping.
func (pn *PerlinNoise) AlpineRidgeNoise(x, y, z float64, octaves int, persistence, lacunarity float64) float64 {
	total := 0.0
	frequency := 1.0
	amplitude := 1.0
	weight := 1.0
	maxValue := 0.0

	for i := 0; i < octaves; i++ {
		n := 1.0 - math.Abs(pn.Noise3D(x*frequency, y*frequency, z*frequency))
		n = n * n * n // Cube for ultra-sharp knife edges
		n *= weight
		weight = math.Max(0.0, math.Min(1.0, n*2.0)) // Successive octaves concentrate along peak crests

		total += n * amplitude
		maxValue += amplitude
		amplitude *= persistence
		frequency *= lacunarity
	}

	return total / maxValue
}

// BillowNoise creates rounded hills, dome formations, and rolling plains.
func (pn *PerlinNoise) BillowNoise(x, y, z float64, octaves int, persistence, lacunarity float64) float64 {
	total := 0.0
	frequency := 1.0
	amplitude := 1.0
	maxValue := 0.0

	for i := 0; i < octaves; i++ {
		n := math.Abs(pn.Noise3D(x*frequency, y*frequency, z*frequency))*2.0 - 1.0
		total += n * amplitude
		maxValue += amplitude
		amplitude *= persistence
		frequency *= lacunarity
	}

	return (total/maxValue + 1.0) * 0.5
}

// DomainWarp applies procedural coordinates displacement to create natural meandering terrain curves.
func (pn *PerlinNoise) DomainWarp(x, y, z float64, warpStrength float64) (float64, float64, float64) {
	qX := pn.Noise3D(x+1.7, y+9.2, z+0.3)
	qY := pn.Noise3D(x+8.3, y+2.8, z+4.1)
	qZ := pn.Noise3D(x+4.5, y+5.6, z+7.9)

	rX := pn.Noise3D(x+qX*warpStrength+1.3, y+qY*warpStrength+2.7, z+qZ*warpStrength+5.1)
	rY := pn.Noise3D(x+qX*warpStrength+3.9, y+qY*warpStrength+6.2, z+qZ*warpStrength+1.4)
	rZ := pn.Noise3D(x+qX*warpStrength+7.1, y+qY*warpStrength+4.8, z+qZ*warpStrength+8.6)

	return x + rX*warpStrength, y + rY*warpStrength, z + rZ*warpStrength
}

func fade(t float64) float64 {
	return t * t * t * (t*(t*6-15) + 10)
}

func lerp(t, a, b float64) float64 {
	return a + t*(b-a)
}

func grad(hash int, x, y, z float64) float64 {
	h := hash & 15
	u := x
	if h >= 8 {
		u = y
	}
	v := y
	if h >= 4 {
		if h == 12 || h == 14 {
			v = x
		} else {
			v = z
		}
	}
	res := 0.0
	if (h & 1) != 0 {
		res -= u
	} else {
		res += u
	}
	if (h & 2) != 0 {
		res -= v
	} else {
		res += v
	}
	return res
}
