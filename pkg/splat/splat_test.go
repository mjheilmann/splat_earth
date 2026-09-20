package splat

import (
	"bytes"
	"math"
	"testing"
)

func TestSplatBinaryRoundTrip(t *testing.T) {
	original := []Splat{
		{
			Position: [3]float32{10.5, -20.25, 30.75},
			Scale:    [3]float32{1.2, 2.4, 0.1},
			Rotation: [4]float32{0.7071, 0.0, 0.7071, 0.0},
			Color:    [4]uint8{255, 128, 64, 200},
		},
		{
			Position: [3]float32{0.0, 0.0, 0.0},
			Scale:    [3]float32{0.5, 0.5, 0.5},
			Rotation: [4]float32{1.0, 0.0, 0.0, 0.0},
			Color:    [4]uint8{0, 255, 0, 255},
		},
	}

	var buf bytes.Buffer
	if err := WriteSplat(&buf, original); err != nil {
		t.Fatalf("WriteSplat failed: %v", err)
	}

	if buf.Len() != len(original)*SplatByteSize {
		t.Fatalf("Buffer size mismatch: got %d, want %d", buf.Len(), len(original)*SplatByteSize)
	}

	read, err := ReadSplat(&buf)
	if err != nil {
		t.Fatalf("ReadSplat failed: %v", err)
	}

	if len(read) != len(original) {
		t.Fatalf("Splat count mismatch: got %d, want %d", len(read), len(original))
	}

	for i := range original {
		orig := original[i]
		r := read[i]

		for j := 0; j < 3; j++ {
			if orig.Position[j] != r.Position[j] {
				t.Errorf("[%d] Position[%d] mismatch: got %v, want %v", i, j, r.Position[j], orig.Position[j])
			}
			if orig.Scale[j] != r.Scale[j] {
				t.Errorf("[%d] Scale[%d] mismatch: got %v, want %v", i, j, r.Scale[j], orig.Scale[j])
			}
		}

		for j := 0; j < 4; j++ {
			if orig.Color[j] != r.Color[j] {
				t.Errorf("[%d] Color[%d] mismatch: got %v, want %v", i, j, r.Color[j], orig.Color[j])
			}
			// Quantized rotation within 1/128 tolerance
			if math.Abs(float64(orig.Rotation[j]-r.Rotation[j])) > 0.02 {
				t.Errorf("[%d] Rotation[%d] mismatch: got %v, want %v", i, j, r.Rotation[j], orig.Rotation[j])
			}
		}
	}
}

func TestWritePLY(t *testing.T) {
	splats := []Splat{
		{
			Position: [3]float32{1, 2, 3},
			Scale:    [3]float32{0.5, 0.5, 0.1},
			Rotation: [4]float32{1, 0, 0, 0},
			Color:    [4]uint8{200, 100, 50, 255},
		},
	}

	var buf bytes.Buffer
	if err := WritePLY(&buf, splats); err != nil {
		t.Fatalf("WritePLY failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("PLY output is empty")
	}
}
