package terrain

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultBiomeConfig(t *testing.T) {
	cfg := DefaultBiomeConfig()
	if len(cfg.LandBands) == 0 {
		t.Fatalf("expected non-empty LandBands")
	}
	if len(cfg.BathymetryBands) == 0 {
		t.Fatalf("expected non-empty BathymetryBands")
	}
	if cfg.EquatorSnowlineM <= 0 {
		t.Errorf("expected positive EquatorSnowlineM, got %f", cfg.EquatorSnowlineM)
	}
}

func TestBiomeConfigLoadSave(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_biomes.json")

	cfg := DefaultBiomeConfig()
	cfg.EquatorSnowlineM = 5200.0

	err := SaveBiomeConfigFile(filePath, cfg)
	if err != nil {
		t.Fatalf("failed to save biome config: %v", err)
	}

	loaded, err := LoadBiomeConfigFile(filePath)
	if err != nil {
		t.Fatalf("failed to load biome config: %v", err)
	}

	if loaded.EquatorSnowlineM != 5200.0 {
		t.Errorf("expected EquatorSnowlineM=5200.0, got %f", loaded.EquatorSnowlineM)
	}
	if len(loaded.LandBands) != len(cfg.LandBands) {
		t.Errorf("expected %d land bands, got %d", len(cfg.LandBands), len(loaded.LandBands))
	}
}

func TestSampleBiomeColorPhysicalElevations(t *testing.T) {
	cfg := DefaultBiomeConfig()
	seaLevel := cfg.SeaLevel
	landSpan := 1.0 - seaLevel

	// 1. Water (50% depth)
	colWater := cfg.SampleBiomeColor(seaLevel, 0.0, 0.0, seaLevel*0.5, 0.0, 0.0)
	if colWater[2] <= colWater[0] { // Blue should exceed Red
		t.Errorf("expected ocean color to be blue-dominated, got RGB: %v", colWater)
	}

	// 2. Beach shoreline (1m physical elevation)
	elevBeach := seaLevel + (1.0/8848.0)*landSpan
	colBeach := cfg.SampleBiomeColor(seaLevel, 28.0, -82.0, elevBeach, 0.0, 0.0)
	if colBeach[0] < 100 { // Sand/light beach tint
		t.Errorf("expected beach sand tint at 1m elevation, got RGB: %v", colBeach)
	}

	// 3. Florida / Humid Lowlands (50m physical elevation)
	elevFlorida := seaLevel + (50.0/8848.0)*landSpan
	colFlorida := cfg.SampleBiomeColor(seaLevel, 28.0, -82.0, elevFlorida, 0.0, 0.0)
	// Emerald green: Green channel should be highest
	if colFlorida[1] <= colFlorida[0] || colFlorida[1] <= colFlorida[2] {
		t.Errorf("expected Florida at 50m to be lush green, got RGB: %v", colFlorida)
	}

	// 4. Great Plains (1000m physical elevation)
	elevPlains := seaLevel + (1000.0/8848.0)*landSpan
	colPlains := cfg.SampleBiomeColor(seaLevel, 40.0, -100.0, elevPlains, 0.0, 0.0)
	// Prairie golden/ochre tones
	if colPlains[0] < colFlorida[0] {
		t.Errorf("expected Great Plains to have warm/amber tones higher than deep green lowlands, got %v vs %v", colPlains, colFlorida)
	}

	// 5. Alpine Snowcap (5000m physical elevation at equator)
	elevSnow := seaLevel + (5000.0/8848.0)*landSpan
	colSnow := cfg.SampleBiomeColor(seaLevel, 0.0, -78.0, elevSnow, 0.0, 0.0)
	if colSnow[0] < 200 || colSnow[1] < 200 || colSnow[2] < 200 {
		t.Errorf("expected white snowcap at 5000m, got RGB: %v", colSnow)
	}
}

func TestFindDefaultBiomeConfig(t *testing.T) {
	// Verify FindDefaultBiomeConfig finds config/biomes.json
	path := FindDefaultBiomeConfig()
	if path == "" {
		t.Log("Note: FindDefaultBiomeConfig returned empty from test dir (normal if running from subpkg)")
	} else {
		if _, err := os.Stat(path); err != nil {
			t.Errorf("returned path %s does not exist: %v", path, err)
		}
	}
}
