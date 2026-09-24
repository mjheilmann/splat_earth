package splat

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync"
)

// MagicIdentifier for Compressed Indexed Splat Container format (.splat / .csplat)
var ContainerMagic = [8]byte{'C', 'S', 'P', 'L', 'A', 'T', 0x01, 0x00}

// ContainerHeader holds top-level spheroid and terrain metadata.
type ContainerHeader struct {
	Magic            [8]byte
	Version          uint32
	Radius           float64
	Flattening       float64
	TotalSplatCount  uint64
	VirtualPoints    uint64
	MeanSpacingKm    float64
	ChunkCount       uint32
	MaxLODLevels     uint32
	ManifestOffset   uint64
	ManifestByteSize uint64
}

// ChunkLODEntry holds disk offset and size information for a single chunk LOD level.
type ChunkLODEntry struct {
	LODLevel         uint32
	SplatCount       uint32
	FileOffset       uint64
	CompressedSize   uint64
	UncompressedSize uint64
}

// ChunkManifest describes a spatial chunk bounding region and its LOD table.
type ChunkManifest struct {
	ChunkID uint32
	Center  [3]float32
	Radius  float32
	Normal  [3]float32
	LatMin  float32
	LatMax  float32
	LonMin  float32
	LonMax  float32
	LODs    []ChunkLODEntry
}

// ContainerReader provides random-access streaming of compressed chunk LODs directly from disk.
type ContainerReader struct {
	file     *os.File
	header   ContainerHeader
	chunks   []ChunkManifest
	chunkMap map[uint32]*ChunkManifest
	mu       sync.RWMutex
}

// OpenContainer opens an indexed splat container file and reads its manifest header into memory.
func OpenContainer(filePath string) (*ContainerReader, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open container file %s: %w", filePath, err)
	}

	var header ContainerHeader
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read container header: %w", err)
	}

	if header.Magic != ContainerMagic {
		f.Close()
		return nil, fmt.Errorf("invalid container magic: %v (expected %v)", header.Magic, ContainerMagic)
	}

	// Seek to manifest table
	if _, err := f.Seek(int64(header.ManifestOffset), io.SeekStart); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to seek to manifest table at offset %d: %w", header.ManifestOffset, err)
	}

	// Read compressed manifest buffer
	compManifestBuf := make([]byte, header.ManifestByteSize)
	if _, err := io.ReadFull(f, compManifestBuf); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read compressed manifest table: %w", err)
	}

	// Decompress manifest
	zr, err := zlib.NewReader(bytes.NewReader(compManifestBuf))
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to decompress manifest table: %w", err)
	}
	defer zr.Close()

	chunks := make([]ChunkManifest, header.ChunkCount)
	chunkMap := make(map[uint32]*ChunkManifest, header.ChunkCount)

	for i := uint32(0); i < header.ChunkCount; i++ {
		var cm ChunkManifest
		if err := binary.Read(zr, binary.LittleEndian, &cm.ChunkID); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk ID: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.Center); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk Center: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.Radius); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk Radius: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.Normal); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk Normal: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.LatMin); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk LatMin: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.LatMax); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk LatMax: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.LonMin); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk LonMin: %w", err)
		}
		if err := binary.Read(zr, binary.LittleEndian, &cm.LonMax); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk LonMax: %w", err)
		}

		var lodCount uint32
		if err := binary.Read(zr, binary.LittleEndian, &lodCount); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read chunk LOD count: %w", err)
		}

		cm.LODs = make([]ChunkLODEntry, lodCount)
		for j := uint32(0); j < lodCount; j++ {
			if err := binary.Read(zr, binary.LittleEndian, &cm.LODs[j]); err != nil {
				f.Close()
				return nil, fmt.Errorf("failed to read chunk LOD entry: %w", err)
			}
		}

		chunks[i] = cm
		chunkMap[cm.ChunkID] = &chunks[i]
	}

	return &ContainerReader{
		file:     f,
		header:   header,
		chunks:   chunks,
		chunkMap: chunkMap,
	}, nil
}

// Close closes the underlying container file.
func (cr *ContainerReader) Close() error {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if cr.file != nil {
		return cr.file.Close()
	}
	return nil
}

// Header returns a copy of the container header.
func (cr *ContainerReader) Header() ContainerHeader {
	return cr.header
}

// Chunks returns a slice of all chunk manifests.
func (cr *ContainerReader) Chunks() []ChunkManifest {
	return cr.chunks
}

// ReadChunkLODBytes reads and decompresses raw binary splat data for a chunk and LOD level directly from disk.
func (cr *ContainerReader) ReadChunkLODBytes(chunkID uint32, lodLevel uint32) ([]byte, error) {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	cm, exists := cr.chunkMap[chunkID]
	if !exists {
		return nil, fmt.Errorf("chunk ID %d not found", chunkID)
	}

	var targetLOD *ChunkLODEntry
	for i := range cm.LODs {
		if cm.LODs[i].LODLevel == lodLevel {
			targetLOD = &cm.LODs[i]
			break
		}
	}

	if targetLOD == nil {
		// Fallback to highest available LOD
		if len(cm.LODs) > 0 {
			targetLOD = &cm.LODs[len(cm.LODs)-1]
		} else {
			return nil, fmt.Errorf("no LOD data available for chunk ID %d", chunkID)
		}
	}

	if targetLOD.CompressedSize == 0 || targetLOD.UncompressedSize == 0 {
		return []byte{}, nil
	}

	compBuf := make([]byte, targetLOD.CompressedSize)
	if _, err := cr.file.ReadAt(compBuf, int64(targetLOD.FileOffset)); err != nil {
		return nil, fmt.Errorf("failed to read chunk bytes at offset %d: %w", targetLOD.FileOffset, err)
	}

	zr, err := zlib.NewReader(bytes.NewReader(compBuf))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize zlib decompressor: %w", err)
	}
	defer zr.Close()

	uncompBuf := make([]byte, targetLOD.UncompressedSize)
	if _, err := io.ReadFull(zr, uncompBuf); err != nil {
		return nil, fmt.Errorf("failed to decompress chunk data: %w", err)
	}

	return uncompBuf, nil
}

// ReadChunkLODSplats reads and parses splats for a chunk and LOD level.
func (cr *ContainerReader) ReadChunkLODSplats(chunkID uint32, lodLevel uint32) ([]Splat, error) {
	buf, err := cr.ReadChunkLODBytes(chunkID, lodLevel)
	if err != nil {
		return nil, err
	}
	return ReadSplat(bytes.NewReader(buf))
}

// IsContainerFile checks if a file begins with the container magic bytes.
func IsContainerFile(filePath string) bool {
	f, err := os.Open(filePath)
	if err != nil {
		return false
	}
	defer f.Close()

	var magic [8]byte
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return false
	}
	return magic == ContainerMagic
}
