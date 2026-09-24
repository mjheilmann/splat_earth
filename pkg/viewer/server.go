package viewer

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/mjheilmann/gaussian-splat-app/pkg/splat"
	"github.com/mjheilmann/gaussian-splat-app/pkg/terrain"
)

//go:embed embedded/*
var embeddedFS embed.FS

// Server handles serving the 3D Gaussian splat viewer, chunk manifests, and multi-LOD splat streams.
type Server struct {
	mu             sync.RWMutex
	filePath       string
	splats         []splat.Splat
	splatSet       *splat.SplatSet
	chunkedTerrain *terrain.ChunkedTerrain
	proceduralCube *terrain.ProceduralCubedSphere
	chunkCache     map[string][]splat.Splat
	cacheOrder     []string
	spheroidInfo   struct {
		Radius     float64 `json:"radius"`
		Flattening float64 `json:"flattening"`
	}
}

// NewServer initializes a viewer server with a splat file path or heightmap data source.
func NewServer(filePath string) (*Server, error) {
	s := &Server{
		filePath: filePath,
	}

	if filePath != "" {
		if err := s.loadFile(filePath); err != nil {
			return nil, err
		}
	} else {
		// Generate default initial terrain from heightmap data (<1 MB RAM)
		cfg := terrain.DefaultConfig()
		gen := terrain.NewTerrainGenerator(cfg)
		s.proceduralCube = gen.NewProceduralCubedSphere(4)
		s.chunkCache = make(map[string][]splat.Splat)
		s.spheroidInfo.Radius = cfg.Spheroid.SemiMajorAxis
		s.spheroidInfo.Flattening = cfg.Spheroid.Flattening
	}

	return s, nil
}

func (s *Server) loadFile(path string) error {
	ext := strings.ToLower(filepath.Ext(path))
	var splats []splat.Splat
	var err error

	if ext == ".splat" {
		splats, err = splat.LoadSplatFile(path)
	} else {
		splats, err = splat.LoadSplatFile(path)
	}

	if err != nil {
		return fmt.Errorf("failed to load %s: %w", path, err)
	}

	s.mu.Lock()
	s.filePath = path
	s.splats = splats
	s.splatSet = splat.NewSplatSet(splats)

	// Estimate radius from splat bounds
	maxBound := float64(s.splatSet.Max[0])
	if float64(s.splatSet.Max[1]) > maxBound {
		maxBound = float64(s.splatSet.Max[1])
	}
	if maxBound < 1.0 {
		maxBound = 100.0
	}
	s.spheroidInfo.Radius = maxBound
	s.spheroidInfo.Flattening = 0.05

	// Partition loaded splats into a single chunk for compatibility
	s.chunkedTerrain = nil
	s.proceduralCube = nil
	s.mu.Unlock()

	return nil
}

// Router returns an http.Handler for the viewer service.
func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()

	// API endpoints
	mux.HandleFunc("/api/meta", s.handleMeta)
	mux.HandleFunc("/api/chunks", s.handleChunks)
	mux.HandleFunc("/api/chunk", s.handleChunkBinary)
	mux.HandleFunc("/api/splat", s.handleSplatBinary)

	// Embedded web static assets
	sub, err := fs.Sub(embeddedFS, "embedded")
	if err != nil {
		panic(err)
	}
	mux.Handle("/", http.FileServer(http.FS(sub)))

	return mux
}

func (s *Server) handleMeta(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	totalSplats := 0
	chunkCount := 0
	var boundsMin, boundsMax [3]float32

	if s.proceduralCube != nil {
		chunkCount = s.proceduralCube.TotalChunks
		for _, m := range s.proceduralCube.Manifests {
			totalSplats += m.CountsLOD[0]
		}
		R := float32(s.spheroidInfo.Radius * 1.15)
		boundsMin = [3]float32{-R, -R, -R}
		boundsMax = [3]float32{R, R, R}
	} else if s.chunkedTerrain != nil {
		chunkCount = len(s.chunkedTerrain.Chunks)
		for _, m := range s.chunkedTerrain.Manifests {
			totalSplats += m.CountsLOD[0]
		}
		R := float32(s.spheroidInfo.Radius * 1.15)
		boundsMin = [3]float32{-R, -R, -R}
		boundsMax = [3]float32{R, R, R}
	} else {
		totalSplats = len(s.splats)
		if s.splatSet != nil {
			boundsMin = s.splatSet.Min
			boundsMax = s.splatSet.Max
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-cache")

	var densityInfo interface{}
	var focusInfo interface{}
	var adaptiveDensity bool
	var curvatureTolerance float64
	var virtualPoints int

	if s.proceduralCube != nil {
		cfg := s.proceduralCube.Config
		adaptiveDensity = cfg.AdaptiveDensity
		curvatureTolerance = cfg.CurvatureTolerance
		virtualPoints = cfg.StepsLat * cfg.StepsLon
		focusInfo = cfg.FocusArea

		radiusKm := s.spheroidInfo.Radius
		bKm := radiusKm * (1.0 - s.spheroidInfo.Flattening)
		surfAreaKm2 := 2.0*3.14159265*radiusKm*radiusKm + 3.14159265*(radiusKm*bKm)
		spacingMeters := 0.0
		if totalSplats > 0 {
			spacingMeters = 1000.0 * math.Sqrt(surfAreaKm2/float64(totalSplats))
		}
		densityInfo = map[string]interface{}{
			"totalPoints":               totalSplats,
			"surfaceAreaKm2":            surfAreaKm2,
			"splatsPerKm2":              float64(totalSplats) / surfAreaKm2,
			"meanSpacingMeters":         spacingMeters,
			"recommendedPatchesPerFace": s.proceduralCube.PatchesPerFaceAxis,
			"totalChunks":               s.proceduralCube.TotalChunks,
			"splatsPerChunkLOD0":        totalSplats / s.proceduralCube.TotalChunks,
			"focusEnabled":              cfg.FocusArea.Enabled,
		}
	} else {
		adaptiveDensity = true
		curvatureTolerance = 0.12
		virtualPoints = totalSplats
		radiusKm := s.spheroidInfo.Radius
		bKm := radiusKm * (1.0 - s.spheroidInfo.Flattening)
		surfAreaKm2 := 2.0*3.14159265*radiusKm*radiusKm + 3.14159265*(radiusKm*bKm)
		spacingMeters := 0.0
		if totalSplats > 0 {
			spacingMeters = 1000.0 * math.Sqrt(surfAreaKm2/float64(totalSplats))
		}
		densityInfo = map[string]interface{}{
			"totalPoints":               totalSplats,
			"surfaceAreaKm2":            surfAreaKm2,
			"splatsPerKm2":              float64(totalSplats) / surfAreaKm2,
			"meanSpacingMeters":         spacingMeters,
			"recommendedPatchesPerFace": 2,
			"totalChunks":               24,
			"splatsPerChunkLOD0":        totalSplats / 24,
			"focusEnabled":              false,
		}
		focusInfo = map[string]interface{}{
			"enabled":           false,
			"centerLat":         0.0,
			"centerLon":         0.0,
			"areaSqMiles":       0.0,
			"densityMultiplier": 0.0,
			"transitionWidthKm": 0.0,
		}
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"splatCount":         totalSplats,
		"virtualPoints":      virtualPoints,
		"chunkCount":         chunkCount,
		"adaptiveDensity":    adaptiveDensity,
		"curvatureTolerance": curvatureTolerance,
		"density":            densityInfo,
		"focusArea":          focusInfo,
		"byteSize":           totalSplats * 32,
		"bounds": map[string]interface{}{
			"min": boundsMin,
			"max": boundsMax,
		},
		"spheroid": map[string]interface{}{
			"radius":     s.spheroidInfo.Radius,
			"flattening": s.spheroidInfo.Flattening,
		},
	})
}

func (s *Server) handleChunks(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-cache")

	if s.proceduralCube != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"chunks": s.proceduralCube.Manifests,
		})
		return
	}

	if s.chunkedTerrain != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"chunks": s.chunkedTerrain.Manifests,
		})
		return
	}

	// Fallback for flat non-chunked splat files: expose 1 single virtual chunk covering entire model
	radius := float32(s.spheroidInfo.Radius * 1.5)
	counts := [5]int{len(s.splats), len(s.splats), len(s.splats), len(s.splats), len(s.splats)}
	fallbackManifests := []terrain.ChunkManifest{
		{
			ID:        0,
			Center:    [3]float32{0, 0, 0},
			Radius:    radius,
			Normal:    [3]float32{0, 0, 1},
			CountsLOD: counts,
		},
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"chunks": fallbackManifests,
	})
}

func (s *Server) handleChunkBinary(w http.ResponseWriter, r *http.Request) {
	chunkIDStr := r.URL.Query().Get("id")
	lodStr := r.URL.Query().Get("lod")

	chunkID, err := strconv.Atoi(chunkIDStr)
	if err != nil {
		http.Error(w, "Invalid chunk ID", http.StatusBadRequest)
		return
	}

	lod, err := strconv.Atoi(lodStr)
	if err != nil || lod < 0 || lod > 4 {
		lod = 0
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "no-cache")

	s.mu.RLock()
	procCube := s.proceduralCube
	chunked := s.chunkedTerrain
	s.mu.RUnlock()

	// On-Demand Procedural Chunk Synthesis with LRU Cache (< 30 MB server RAM)
	if procCube != nil && chunkID >= 0 && chunkID < procCube.TotalChunks {
		cacheKey := fmt.Sprintf("%d_%d", chunkID, lod)
		s.mu.Lock()
		if s.chunkCache == nil {
			s.chunkCache = make(map[string][]splat.Splat)
		}
		splats, found := s.chunkCache[cacheKey]
		if !found {
			splats = procCube.SampleChunkLOD(chunkID, lod)
			if len(s.chunkCache) > 256 {
				if len(s.cacheOrder) > 0 {
					delete(s.chunkCache, s.cacheOrder[0])
					s.cacheOrder = s.cacheOrder[1:]
				}
			}
			s.chunkCache[cacheKey] = splats
			s.cacheOrder = append(s.cacheOrder, cacheKey)
		}
		s.mu.Unlock()

		_ = splat.WriteSplat(w, splats)
		return
	}

	if chunked != nil && chunkID >= 0 && chunkID < len(chunked.Chunks) {
		splats := chunked.Chunks[chunkID].LODs[lod]
		_ = splat.WriteSplat(w, splats)
		return
	}

	// Fallback to full splats
	s.mu.RLock()
	defer s.mu.RUnlock()
	_ = splat.WriteSplat(w, s.splats)
}

func (s *Server) handleSplatBinary(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", "inline; filename=\"terrain.splat\"")
	w.Header().Set("Cache-Control", "no-cache")

	if s.proceduralCube != nil {
		for id := 0; id < s.proceduralCube.TotalChunks; id++ {
			splats := s.proceduralCube.SampleChunkLOD(id, 0)
			if err := splat.WriteSplat(w, splats); err != nil {
				return
			}
		}
		return
	}

	if s.chunkedTerrain != nil {
		for _, ch := range s.chunkedTerrain.Chunks {
			if len(ch.LODs) > 0 {
				if err := splat.WriteSplat(w, ch.LODs[0]); err != nil {
					return
				}
			}
		}
		return
	}

	_ = splat.WriteSplat(w, s.splats)
}
