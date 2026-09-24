package splat

import (
	"path/filepath"
	"testing"
)

func TestContainerRoundTrip(t *testing.T) {
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "test.csplat")

	header := ContainerHeader{
		Radius:          100.0,
		Flattening:      0.05,
		TotalSplatCount: 200,
		VirtualPoints:   2000,
		MeanSpacingKm:   1.25,
		MaxLODLevels:    3,
	}

	cw, err := NewContainerWriter(filePath, header)
	if err != nil {
		t.Fatalf("NewContainerWriter failed: %v", err)
	}

	// Create test chunks
	for chunkID := uint32(0); chunkID < 2; chunkID++ {
		cm := ChunkManifest{
			ChunkID: chunkID,
			Center:  [3]float32{float32(chunkID * 10), 0, 100},
			Radius:  15.0,
			Normal:  [3]float32{0, 0, 1},
			LatMin:  -10.0,
			LatMax:  10.0,
			LonMin:  -20.0,
			LonMax:  20.0,
			LODs:    make([]ChunkLODEntry, 0),
		}

		for lod := uint32(0); lod < 3; lod++ {
			splats := []Splat{
				{
					Position: [3]float32{float32(chunkID*10 + lod), 0, 100},
					Scale:    [3]float32{1.0, 1.0, 0.1},
					Rotation: [4]float32{1.0, 0, 0, 0},
					Color:    [4]uint8{200, 150, 100, 255},
				},
				{
					Position: [3]float32{float32(chunkID*10 + lod + 1), 1, 100},
					Scale:    [3]float32{1.0, 1.0, 0.1},
					Rotation: [4]float32{1.0, 0, 0, 0},
					Color:    [4]uint8{100, 200, 100, 255},
				},
			}

			entry, err := cw.WriteChunkLOD(int(chunkID), lod, splats)
			if err != nil {
				t.Fatalf("WriteChunkLOD failed: %v", err)
			}
			cm.LODs = append(cm.LODs, *entry)
		}

		cw.AddChunkManifest(cm)
	}

	if err := cw.Finalize(); err != nil {
		t.Fatalf("Finalize failed: %v", err)
	}

	// Check IsContainerFile
	if !IsContainerFile(filePath) {
		t.Fatalf("IsContainerFile returned false for valid container")
	}

	// Read container back
	cr, err := OpenContainer(filePath)
	if err != nil {
		t.Fatalf("OpenContainer failed: %v", err)
	}
	defer cr.Close()

	if cr.Header().Radius != 100.0 {
		t.Errorf("expected radius 100.0, got %f", cr.Header().Radius)
	}
	if len(cr.Chunks()) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(cr.Chunks()))
	}

	// Test reading chunk LOD splats
	splats, err := cr.ReadChunkLODSplats(1, 0)
	if err != nil {
		t.Fatalf("ReadChunkLODSplats failed: %v", err)
	}
	if len(splats) != 2 {
		t.Fatalf("expected 2 splats, got %d", len(splats))
	}
	if splats[0].Position[0] != 10.0 {
		t.Errorf("expected position X 10.0, got %f", splats[0].Position[0])
	}
}
