package terrain

import (
	"encoding/json"
	"math"
	"os"
)

// BiomeElevationBand defines a user-tunable terrestrial elevation band.
type BiomeElevationBand struct {
	Name       string   `json:"name"`        // Human-readable identifier (e.g. "Coastal Lowlands")
	MaxElevM   float64  `json:"max_elev_m"`  // Upper elevation ceiling in meters (0m - 8848m)
	ColorStart [3]uint8 `json:"color_start"` // RGB at the lower bound of this band [0-255, 0-255, 0-255]
	ColorEnd   [3]uint8 `json:"color_end"`   // RGB at the upper bound of this band [0-255, 0-255, 0-255]
}

// BathymetryBand defines a user-tunable underwater depth band.
type BathymetryBand struct {
	Name         string   `json:"name"`           // Identifier (e.g. "Deep Trench", "Continental Shelf")
	MaxDepthFrac float64  `json:"max_depth_frac"` // Depth fraction from 0.0 (Mariana Trench -11000m) to 1.0 (Sea Level 0m)
	ColorStart   [3]uint8 `json:"color_start"`    // RGB at band bottom
	ColorEnd     [3]uint8 `json:"color_end"`      // RGB at band top
}

// BiomeConfig contains all user-tunable elevation boundaries, color gradients, and latitude/slope rules.
type BiomeConfig struct {
	SeaLevel            float64              `json:"sea_level"`            // Normalized sea level threshold (e.g. 0.5622 for 143.35/255)
	EquatorSnowlineM    float64              `json:"equator_snowline_m"`   // Snowline altitude at Equator (e.g. 4800m)
	PoleSnowlineM       float64              `json:"pole_snowline_m"`      // Snowline altitude at Poles (e.g. 0m)
	SnowColor           [3]uint8             `json:"snow_color"`           // RGB for glacial ice & snowcaps
	CliffSlopeThreshold float64              `json:"cliff_slope_threshold"`// Slope gradient (dh/dx) where rock exposed (e.g. 0.28 = ~16 degrees)
	CliffColor          [3]uint8             `json:"cliff_color"`          // RGB for exposed granite / cliff bedrock
	DesertBeltEnabled   bool                 `json:"desert_belt_enabled"`  // Enable subtropical horse-latitude desert tinting
	BathymetryBands     []BathymetryBand     `json:"bathymetry_bands"`     // Underwater depth tiers
	LandBands           []BiomeElevationBand `json:"land_bands"`           // Terrestrial hypsometric elevation tiers
}

// DefaultBiomeConfig returns satellite-realistic default elevation bands and color palettes.
func DefaultBiomeConfig() BiomeConfig {
	return BiomeConfig{
		SeaLevel:            143.35 / 255.0, // ~0.56215 - cleanly separates shallow shelf (143) from land (>= 143.5)
		EquatorSnowlineM:    4800.0,
		PoleSnowlineM:       0.0,
		SnowColor:           [3]uint8{242, 248, 255},
		CliffSlopeThreshold: 0.28,
		CliffColor:          [3]uint8{74, 72, 70},
		DesertBeltEnabled:   true,
		BathymetryBands: []BathymetryBand{
			{
				Name:         "Abyssal Trench & Deep Ocean (-11000m to -4000m)",
				MaxDepthFrac: 0.60,
				ColorStart:   [3]uint8{10, 28, 65},
				ColorEnd:     [3]uint8{18, 55, 120},
			},
			{
				Name:         "Continental Slope & Mid-Depth (-4000m to -500m)",
				MaxDepthFrac: 0.92,
				ColorStart:   [3]uint8{18, 55, 120},
				ColorEnd:     [3]uint8{28, 95, 155},
			},
			{
				Name:         "Coastal Shelf & Reefs (-500m to 0m)",
				MaxDepthFrac: 1.00,
				ColorStart:   [3]uint8{28, 95, 155},
				ColorEnd:     [3]uint8{48, 148, 185},
			},
		},
		LandBands: []BiomeElevationBand{
			{
				Name:       "Immediate Shoreline & Beach Fringe (0m - 12m)",
				MaxElevM:   12.0,
				ColorStart: [3]uint8{215, 202, 152}, // Sand at water boundary
				ColorEnd:   [3]uint8{38, 118, 48},   // Blends into coastal greenery
			},
			{
				Name:       "Humid Coastal Lowlands & Subtropics (12m - 180m)",
				MaxElevM:   180.0,
				ColorStart: [3]uint8{38, 118, 48},  // Florida Everglades & coastal forest
				ColorEnd:   [3]uint8{48, 132, 52},
			},
			{
				Name:       "Temperate Interior Lowlands & Farmland (180m - 450m)",
				MaxElevM:   450.0,
				ColorStart: [3]uint8{48, 132, 52},  // Midwest / Ohio Valley
				ColorEnd:   [3]uint8{78, 138, 58},
			},
			{
				Name:       "Continental Great Plains & Prairies (450m - 1500m)",
				MaxElevM:   1500.0,
				ColorStart: [3]uint8{78, 138, 58},   // Golden-green prairie
				ColorEnd:   [3]uint8{172, 152, 92},  // Amber-ochre high plains
			},
			{
				Name:       "High Continental Plateaus & Intermountain Basins (1500m - 2500m)",
				MaxElevM:   2500.0,
				ColorStart: [3]uint8{172, 152, 92},  // Tawny sandstone
				ColorEnd:   [3]uint8{156, 128, 88},  // Sagebrush / scrubland
			},
			{
				Name:       "Subalpine Highlands & Montane Conifers (2500m - 3800m)",
				MaxElevM:   3800.0,
				ColorStart: [3]uint8{156, 128, 88},
				ColorEnd:   [3]uint8{115, 112, 105}, // Mountain slate / timberline
			},
			{
				Name:       "High Alpine Ridges & Scree (3800m - 8848m)",
				MaxElevM:   8848.0,
				ColorStart: [3]uint8{115, 112, 105},
				ColorEnd:   [3]uint8{175, 172, 170},
			},
		},
	}
}

// FindDefaultBiomeConfig searches standard directory locations for an optional user-defined biomes.json.
func FindDefaultBiomeConfig() string {
	candidates := []string{
		"config/biomes.json",
		"biomes.json",
		"../config/biomes.json",
		"../biomes.json",
		"../../config/biomes.json",
		"../../biomes.json",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// LoadBiomeConfigFile parses a JSON biome configuration file into a BiomeConfig struct.
func LoadBiomeConfigFile(filePath string) (BiomeConfig, error) {
	cfg := DefaultBiomeConfig()
	data, err := os.ReadFile(filePath)
	if err != nil {
		return cfg, err
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// SaveBiomeConfigFile writes a BiomeConfig struct formatted as pretty JSON.
func SaveBiomeConfigFile(filePath string, cfg BiomeConfig) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

// SampleBiomeColor evaluates the configured color bands given physical coordinates, elevation, and slope.
func (b *BiomeConfig) SampleBiomeColor(seaLevel float64, lat, lon, elev, slopeE, slopeN float64) [4]uint8 {
	elev = clampFloat(elev, 0.0, 1.0)
	slopeMag := math.Sqrt(slopeE*slopeE + slopeN*slopeN)
	absLat := math.Abs(lat)

	effectiveSea := seaLevel
	if b.SeaLevel > 0 {
		effectiveSea = b.SeaLevel
	}

	var r, g, bCol float64
	alpha := uint8(255)

	if elev < effectiveSea {
		// -------------------------------------------------------------
		// Underwater Bathymetry
		// -------------------------------------------------------------
		tDepth := elev / math.Max(0.001, effectiveSea) // 0.0 (Mariana) -> 1.0 (Sea Level)
		bands := b.BathymetryBands
		if len(bands) == 0 {
			bands = DefaultBiomeConfig().BathymetryBands
		}

		prevMax := 0.0
		matched := false
		for _, band := range bands {
			if tDepth <= band.MaxDepthFrac || band == bands[len(bands)-1] {
				span := math.Max(0.001, band.MaxDepthFrac-prevMax)
				t := clampFloat((tDepth-prevMax)/span, 0.0, 1.0)
				r = lerpFloat(t, float64(band.ColorStart[0]), float64(band.ColorEnd[0]))
				g = lerpFloat(t, float64(band.ColorStart[1]), float64(band.ColorEnd[1]))
				bCol = lerpFloat(t, float64(band.ColorStart[2]), float64(band.ColorEnd[2]))
				matched = true
				break
			}
			prevMax = band.MaxDepthFrac
		}
		if !matched {
			r, g, bCol = 20, 60, 130
		}
	} else {
		// -------------------------------------------------------------
		// Terrestrial Land (Physical elevation: 0m -> 8848m)
		// -------------------------------------------------------------
		landSpan := math.Max(0.05, 1.0-effectiveSea)
		elevMeters := ((elev - effectiveSea) / landSpan) * 8848.0

		// Dynamic latitude-dependent snowline
		latFrac := clampFloat(absLat/90.0, 0.0, 1.0)
		eqSnow := b.EquatorSnowlineM
		if eqSnow <= 0 {
			eqSnow = 4800.0
		}
		snowlineMeters := eqSnow - (eqSnow-b.PoleSnowlineM)*math.Pow(latFrac, 1.4)
		if snowlineMeters < 0 {
			snowlineMeters = 0
		}

		if elevMeters >= snowlineMeters {
			// Glacial Snow & Icecap
			t := clampFloat((elevMeters-snowlineMeters)/600.0, 0.0, 1.0)
			snow := b.SnowColor
			if snow[0] == 0 && snow[1] == 0 && snow[2] == 0 {
				snow = [3]uint8{242, 248, 255}
			}
			r = lerpFloat(t, 210, float64(snow[0]))
			g = lerpFloat(t, 220, float64(snow[1]))
			bCol = lerpFloat(t, float64(snow[2])-20, float64(snow[2]))
		} else {
			bands := b.LandBands
			if len(bands) == 0 {
				bands = DefaultBiomeConfig().LandBands
			}

			prevMax := 0.0
			matched := false
			for i, band := range bands {
				if elevMeters <= band.MaxElevM || i == len(bands)-1 {
					span := math.Max(0.1, band.MaxElevM-prevMax)
					t := clampFloat((elevMeters-prevMax)/span, 0.0, 1.0)
					r = lerpFloat(t, float64(band.ColorStart[0]), float64(band.ColorEnd[0]))
					g = lerpFloat(t, float64(band.ColorStart[1]), float64(band.ColorEnd[1]))
					bCol = lerpFloat(t, float64(band.ColorStart[2]), float64(band.ColorEnd[2]))
					matched = true
					break
				}
				prevMax = band.MaxElevM
			}
			if !matched {
				r, g, bCol = 60, 130, 60
			}

			// Subtropical horse-latitude desert belt tint (latitudes 16° to 36°)
			if b.DesertBeltEnabled && absLat >= 16.0 && absLat <= 36.0 && elevMeters < 2000.0 {
				desertStrength := math.Sin((absLat - 16.0) / 20.0 * math.Pi)
				isAridZone := (lon >= -120.0 && lon <= -100.0) || (lon >= -18.0 && lon <= 65.0) || (lon >= 115.0 && lon <= 150.0 && lat < 0)
				if isAridZone {
					w := desertStrength * 0.55
					r = lerpFloat(w, r, 202)
					g = lerpFloat(w, g, 172)
					bCol = lerpFloat(w, bCol, 125)
				}
			}
		}
	}

	// Steep cliff rock shading
	cliffThreshold := b.CliffSlopeThreshold
	if cliffThreshold <= 0 {
		cliffThreshold = 0.28
	}
	if elev > effectiveSea && slopeMag > cliffThreshold {
		rockWeight := math.Min((slopeMag-cliffThreshold)*1.9, 0.88)
		cliff := b.CliffColor
		if cliff[0] == 0 && cliff[1] == 0 && cliff[2] == 0 {
			cliff = [3]uint8{74, 72, 70}
		}
		r = lerpFloat(rockWeight, r, float64(cliff[0]))
		g = lerpFloat(rockWeight, g, float64(cliff[1]))
		bCol = lerpFloat(rockWeight, bCol, float64(cliff[2]))
	}

	return [4]uint8{
		uint8(clampFloat(r, 0, 255)),
		uint8(clampFloat(g, 0, 255)),
		uint8(clampFloat(bCol, 0, 255)),
		alpha,
	}
}

// SampleSeaSurfaceColor evaluates semi-transparent ocean water surface color based on latitude, longitude, and bed depth.
func (b *BiomeConfig) SampleSeaSurfaceColor(seaLevel float64, lat, lon, bedElevNorm float64) [4]uint8 {
	effectiveSea := seaLevel
	if b.SeaLevel > 0 {
		effectiveSea = b.SeaLevel
	}

	// tDepth: 0.0 (deepest trench/abyss) to 1.0 (water surface / shoreline)
	tDepth := clampFloat(bedElevNorm/math.Max(0.001, effectiveSea), 0.0, 1.0)
	absLat := math.Abs(lat)

	var r, g, bCol float64
	var alpha uint8

	if absLat > 68.0 {
		// Polar icy sea surface
		r = lerpFloat(tDepth, 110, 180)
		g = lerpFloat(tDepth, 160, 220)
		bCol = lerpFloat(tDepth, 210, 245)
		alpha = uint8(155)
	} else if tDepth < 0.70 {
		// Deep Abyssal Ocean & Pelagic Waters: Deep rich sapphire
		t := tDepth / 0.70
		r = lerpFloat(t, 16, 28)
		g = lerpFloat(t, 68, 108)
		bCol = lerpFloat(t, 142, 185)
		alpha = uint8(145) // ~57% opacity
	} else if tDepth < 0.95 {
		// Continental Shelf & Shallow Seas: Vibrant Azure
		t := (tDepth - 0.70) / 0.25
		r = lerpFloat(t, 28, 42)
		g = lerpFloat(t, 108, 165)
		bCol = lerpFloat(t, 185, 218)
		alpha = uint8(130) // ~51% opacity
	} else {
		// Tropical Lagoons, Reefs, Beach Shallows: Crystal Aquamarine / Turquoise
		t := (tDepth - 0.95) / 0.05
		r = lerpFloat(t, 42, 58)
		g = lerpFloat(t, 165, 205)
		bCol = lerpFloat(t, 218, 235)
		alpha = uint8(115) // ~45% translucent, reveals sandy seabed underneath!
	}

	return [4]uint8{
		uint8(clampFloat(r, 0, 255)),
		uint8(clampFloat(g, 0, 255)),
		uint8(clampFloat(bCol, 0, 255)),
		alpha,
	}
}

