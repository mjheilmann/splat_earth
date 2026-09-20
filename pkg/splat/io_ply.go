package splat

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const shC0 = 0.28209479177387814 // SH 0-order coefficient: 1 / (2 * sqrt(pi))

// SavePLYFile exports splats to standard 3D Gaussian Splatting binary PLY format.
func SavePLYFile(filename string, splats []Splat) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create ply file %s: %w", filename, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	if err := WritePLY(writer, splats); err != nil {
		return err
	}
	return writer.Flush()
}

// WritePLY writes splats in standard 3DGS binary PLY format.
func WritePLY(w io.Writer, splats []Splat) error {
	header := fmt.Sprintf(`ply
format binary_little_endian 1.0
element vertex %d
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
`, len(splats))

	if _, err := io.WriteString(w, header); err != nil {
		return fmt.Errorf("failed writing PLY header: %w", err)
	}

	// Each vertex record: 17 floats (68 bytes)
	buf := make([]byte, 17*4)

	for _, s := range splats {
		// Pos (x, y, z)
		putFloat32(buf, 0, s.Position[0])
		putFloat32(buf, 4, s.Position[1])
		putFloat32(buf, 8, s.Position[2])

		// Normals (nx, ny, nz)
		putFloat32(buf, 12, 0)
		putFloat32(buf, 16, 0)
		putFloat32(buf, 20, 0)

		// Color to SH DC: f_dc = (color/255.0 - 0.5) / shC0
		rDC := float32((float64(s.Color[0])/255.0 - 0.5) / shC0)
		gDC := float32((float64(s.Color[1])/255.0 - 0.5) / shC0)
		bDC := float32((float64(s.Color[2])/255.0 - 0.5) / shC0)
		putFloat32(buf, 24, rDC)
		putFloat32(buf, 28, gDC)
		putFloat32(buf, 32, bDC)

		// Opacity to logit: log(alpha / (1 - alpha))
		alpha := float64(s.Color[3]) / 255.0
		if alpha < 0.001 {
			alpha = 0.001
		} else if alpha > 0.999 {
			alpha = 0.999
		}
		opacityLogit := float32(math.Log(alpha / (1.0 - alpha)))
		putFloat32(buf, 36, opacityLogit)

		// Log scales
		for i := 0; i < 3; i++ {
			scaleVal := float64(s.Scale[i])
			if scaleVal < 1e-6 {
				scaleVal = 1e-6
			}
			putFloat32(buf, 40+i*4, float32(math.Log(scaleVal)))
		}

		// Rotations (rot_0=W, rot_1=X, rot_2=Y, rot_3=Z)
		putFloat32(buf, 52, s.Rotation[0])
		putFloat32(buf, 56, s.Rotation[1])
		putFloat32(buf, 60, s.Rotation[2])
		putFloat32(buf, 64, s.Rotation[3])

		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("failed writing PLY vertex: %w", err)
		}
	}

	return nil
}

func putFloat32(buf []byte, offset int, v float32) {
	binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(v))
}
