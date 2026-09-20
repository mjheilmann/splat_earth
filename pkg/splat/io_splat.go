package splat

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const SplatByteSize = 32

// WriteSplat writes a slice of Splats to an io.Writer in standard 32-byte .splat binary format.
func WriteSplat(w io.Writer, splats []Splat) error {
	buf := make([]byte, SplatByteSize)
	for _, s := range splats {
		// Pos X, Y, Z (float32)
		binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(s.Position[0]))
		binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(s.Position[1]))
		binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(s.Position[2]))

		// Scale X, Y, Z (float32)
		binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(s.Scale[0]))
		binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(s.Scale[1]))
		binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(s.Scale[2]))

		// Color RGBA (4 bytes)
		buf[24] = s.Color[0]
		buf[25] = s.Color[1]
		buf[26] = s.Color[2]
		buf[27] = s.Color[3]

		// Rotation (W, X, Y, Z) quantized to uint8: (q * 128) + 128 clamped to 0..255
		for i := 0; i < 4; i++ {
			v := math.Round(float64(s.Rotation[i])*128.0 + 128.0)
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			buf[28+i] = uint8(v)
		}

		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("failed to write splat data: %w", err)
		}
	}
	return nil
}

// SaveSplatFile writes splats to a .splat file.
func SaveSplatFile(filename string, splats []Splat) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create splat file %s: %w", filename, err)
	}
	defer file.Close()

	return WriteSplat(file, splats)
}

// ReadSplat reads splats from an io.Reader in standard 32-byte .splat binary format.
func ReadSplat(r io.Reader) ([]Splat, error) {
	var splats []Splat
	buf := make([]byte, SplatByteSize)

	for {
		_, err := io.ReadFull(r, buf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed reading splat stream: %w", err)
		}

		var s Splat
		s.Position[0] = math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4]))
		s.Position[1] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8]))
		s.Position[2] = math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12]))

		s.Scale[0] = math.Float32frombits(binary.LittleEndian.Uint32(buf[12:16]))
		s.Scale[1] = math.Float32frombits(binary.LittleEndian.Uint32(buf[16:20]))
		s.Scale[2] = math.Float32frombits(binary.LittleEndian.Uint32(buf[20:24]))

		s.Color[0] = buf[24]
		s.Color[1] = buf[25]
		s.Color[2] = buf[26]
		s.Color[3] = buf[27]

		for i := 0; i < 4; i++ {
			s.Rotation[i] = (float32(buf[28+i]) - 128.0) / 128.0
		}

		splats = append(splats, s)
	}

	return splats, nil
}

// LoadSplatFile reads splats from a .splat file.
func LoadSplatFile(filename string) ([]Splat, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open splat file %s: %w", filename, err)
	}
	defer file.Close()

	return ReadSplat(file)
}
