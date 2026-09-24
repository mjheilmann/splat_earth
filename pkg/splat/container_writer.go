package splat

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// ContainerWriter creates an indexed compressed splat container on disk.
type ContainerWriter struct {
	file           *os.File
	header         ContainerHeader
	chunks         []ChunkManifest
	currentOffset  int64
	compressionLvl int
}

// NewContainerWriter initializes a new container writer for a target file path.
func NewContainerWriter(filePath string, header ContainerHeader) (*ContainerWriter, error) {
	f, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create container file %s: %w", filePath, err)
	}

	header.Magic = ContainerMagic
	header.Version = 1

	// Write preliminary header (will be finalized on Close)
	if err := binary.Write(f, binary.LittleEndian, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to write initial header: %w", err)
	}

	headerSize := int64(binary.Size(header))
	return &ContainerWriter{
		file:           f,
		header:         header,
		chunks:         make([]ChunkManifest, 0, header.ChunkCount),
		currentOffset:  headerSize,
		compressionLvl: zlib.DefaultCompression,
	}, nil
}

// WriteChunkLOD writes a compressed splat buffer for a specific chunk and LOD level.
func (cw *ContainerWriter) WriteChunkLOD(chunkIndex int, lodLevel uint32, splats []Splat) (*ChunkLODEntry, error) {
	var rawBuf bytes.Buffer
	if err := WriteSplat(&rawBuf, splats); err != nil {
		return nil, fmt.Errorf("failed to serialize splats to binary: %w", err)
	}

	uncompBytes := rawBuf.Bytes()
	var compBuf bytes.Buffer
	zw, err := zlib.NewWriterLevel(&compBuf, cw.compressionLvl)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize zlib writer: %w", err)
	}

	if _, err := zw.Write(uncompBytes); err != nil {
		return nil, fmt.Errorf("failed to compress chunk splats: %w", err)
	}
	if err := zw.Close(); err != nil {
		return nil, fmt.Errorf("failed to close zlib stream: %w", err)
	}

	compressedBytes := compBuf.Bytes()
	offset := uint64(cw.currentOffset)

	n, err := cw.file.Write(compressedBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to write compressed chunk to file: %w", err)
	}

	cw.currentOffset += int64(n)

	entry := &ChunkLODEntry{
		LODLevel:         lodLevel,
		SplatCount:       uint32(len(splats)),
		FileOffset:       offset,
		CompressedSize:   uint64(n),
		UncompressedSize: uint64(len(uncompBytes)),
	}

	return entry, nil
}

// WriteChunkLODBytes writes an already serialized 32-byte splat byte buffer for a chunk and LOD.
func (cw *ContainerWriter) WriteChunkLODBytes(lodLevel uint32, splatCount uint32, uncompBytes []byte) (*ChunkLODEntry, error) {
	var compBuf bytes.Buffer
	zw, err := zlib.NewWriterLevel(&compBuf, cw.compressionLvl)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize zlib writer: %w", err)
	}

	if _, err := zw.Write(uncompBytes); err != nil {
		return nil, fmt.Errorf("failed to compress chunk splat bytes: %w", err)
	}
	if err := zw.Close(); err != nil {
		return nil, fmt.Errorf("failed to close zlib stream: %w", err)
	}

	compressedBytes := compBuf.Bytes()
	offset := uint64(cw.currentOffset)

	n, err := cw.file.Write(compressedBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to write compressed chunk bytes to file: %w", err)
	}

	cw.currentOffset += int64(n)

	entry := &ChunkLODEntry{
		LODLevel:         lodLevel,
		SplatCount:       splatCount,
		FileOffset:       offset,
		CompressedSize:   uint64(n),
		UncompressedSize: uint64(len(uncompBytes)),
	}

	return entry, nil
}

// AddChunkManifest appends a completed chunk manifest to the container.
func (cw *ContainerWriter) AddChunkManifest(cm ChunkManifest) {
	cw.chunks = append(cw.chunks, cm)
}

// Finalize writes the manifest index table and updates the header at offset 0.
func (cw *ContainerWriter) Finalize() error {
	manifestOffset := uint64(cw.currentOffset)

	// Serialize and compress manifest table
	var rawManifest bytes.Buffer
	for _, cm := range cw.chunks {
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.ChunkID); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.Center); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.Radius); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.Normal); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.LatMin); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.LatMax); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.LonMin); err != nil {
			return err
		}
		if err := binary.Write(&rawManifest, binary.LittleEndian, cm.LonMax); err != nil {
			return err
		}

		lodCount := uint32(len(cm.LODs))
		if err := binary.Write(&rawManifest, binary.LittleEndian, lodCount); err != nil {
			return err
		}
		for _, lod := range cm.LODs {
			if err := binary.Write(&rawManifest, binary.LittleEndian, lod); err != nil {
				return err
			}
		}
	}

	var compManifest bytes.Buffer
	zw := zlib.NewWriter(&compManifest)
	if _, err := zw.Write(rawManifest.Bytes()); err != nil {
		return fmt.Errorf("failed to compress manifest: %w", err)
	}
	if err := zw.Close(); err != nil {
		return fmt.Errorf("failed to close manifest zlib stream: %w", err)
	}

	manifestBytes := compManifest.Bytes()
	n, err := cw.file.Write(manifestBytes)
	if err != nil {
		return fmt.Errorf("failed to write manifest table: %w", err)
	}

	cw.header.ChunkCount = uint32(len(cw.chunks))
	cw.header.ManifestOffset = manifestOffset
	cw.header.ManifestByteSize = uint64(n)

	// Seek back to start and update header
	if _, err := cw.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("failed to seek to header position: %w", err)
	}

	if err := binary.Write(cw.file, binary.LittleEndian, &cw.header); err != nil {
		return fmt.Errorf("failed to write final container header: %w", err)
	}

	return cw.file.Close()
}
