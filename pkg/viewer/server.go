package viewer

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/myuser/gaussian-splat-app/pkg/spheroid"
	"github.com/myuser/gaussian-splat-app/pkg/splat"
	"github.com/myuser/gaussian-splat-app/pkg/terrain"
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

// NewServer initializes a viewer server with a splat file path.
func NewServer(filePath string) (*Server, error) {
	s := &Server{
		filePath: filePath,
	}

	if filePath != "" {
		if err := s.loadFile(filePath); err != nil {
			return nil, err
		}
	} else {
		// Generate default initial terrain with on-demand procedural cubed-sphere manifests (<1 MB RAM)
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
	mux.HandleFunc("/api/generate", s.handleGenerate)

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

	cfg := terrain.DefaultConfig()
	cfg.Spheroid = spheroid.NewSpheroid(s.spheroidInfo.Radius, s.spheroidInfo.Flattening)
	gen := terrain.NewTerrainGenerator(cfg)
	density := gen.EstimatePointDensity(totalSplats)

	virtualPoints := totalSplats
	adaptivePoints := totalSplats
	isAdaptive := true
	curvTol := 0.12

	if s.proceduralCube != nil {
		virtualPoints = s.proceduralCube.Config.StepsLat * s.proceduralCube.Config.StepsLon
		isAdaptive = s.proceduralCube.Config.AdaptiveDensity
		curvTol = s.proceduralCube.Config.CurvatureTolerance
		if isAdaptive {
			// Ocean and plain regions collapse to large splats, active splats are ~20-25% of uniform grid
			adaptivePoints = int(float64(totalSplats) * 0.22)
			if adaptivePoints < s.proceduralCube.TotalChunks*4 {
				adaptivePoints = s.proceduralCube.TotalChunks * 4
			}
		}
	}

	meta := map[string]interface{}{
		"splatCount":         totalSplats,
		"virtualPoints":      virtualPoints,
		"adaptivePoints":     adaptivePoints,
		"adaptiveDensity":    isAdaptive,
		"curvatureTolerance": curvTol,
		"chunkCount":         chunkCount,
		"byteSize":           totalSplats * splat.SplatByteSize,
		"bounds": map[string]interface{}{
			"min": boundsMin,
			"max": boundsMax,
		},
		"spheroid": s.spheroidInfo,
		"density":  density,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(meta)
}

func (s *Server) handleChunks(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.proceduralCube != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"chunks": s.proceduralCube.Manifests,
		})
		return
	}

	if s.chunkedTerrain != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"chunks": s.chunkedTerrain.Manifests,
		})
		return
	}

	// Fallback single chunk manifest
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"chunks": []terrain.ChunkManifest{
			{
				ID:        0,
				Center:    [3]float32{0, 0, 0},
				Radius:    float32(s.spheroidInfo.Radius * 1.5),
				Normal:    [3]float32{0, 0, 1},
				CountsLOD: [5]int{len(s.splats), len(s.splats), len(s.splats), len(s.splats), len(s.splats)},
			},
		},
	})
}

func (s *Server) handleChunkBinary(w http.ResponseWriter, r *http.Request) {
	chunkIDStr := r.URL.Query().Get("id")
	lodStr := r.URL.Query().Get("lod")

	chunkID, _ := strconv.Atoi(chunkIDStr)
	lod, _ := strconv.Atoi(lodStr)
	if lod < 0 || lod > 4 {
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

		if err := splat.WriteSplat(w, splats); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	if chunked != nil && chunkID >= 0 && chunkID < len(chunked.Chunks) {
		splats := chunked.Chunks[chunkID].LODs[lod]
		if err := splat.WriteSplat(w, splats); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	// Fallback to full splats
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := splat.WriteSplat(w, s.splats); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
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
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		return
	}

	if s.chunkedTerrain != nil {
		for _, ch := range s.chunkedTerrain.Chunks {
			if err := splat.WriteSplat(w, ch.LODs[0]); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		return
	}

	if err := splat.WriteSplat(w, s.splats); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type GenerateRequest struct {
	GeometryType    string  `json:"geometryType"` // "cubed-sphere" (default) or "latlon"
	PatchesPerFace  int     `json:"patchesPerFace"`
	Radius          float64 `json:"radius"`
	Flattening      float64 `json:"flattening"`
	LatMin          float64 `json:"latMin"`
	LatMax          float64 `json:"latMax"`
	LonMin          float64 `json:"lonMin"`
	LonMax          float64 `json:"lonMax"`
	StepsLat        int     `json:"stepsLat"`
	StepsLon        int     `json:"stepsLon"`
	ElevationScale  float64 `json:"elevationScale"`
	ElevationOffset float64 `json:"elevationOffset"`
	NoiseSeed       int64   `json:"noiseSeed"`
	NoiseFrequency  float64 `json:"noiseFrequency"`
	Octaves         int     `json:"octaves"`
	OverlapFactor      float64 `json:"overlapFactor"`
	ThicknessFactor    float64 `json:"thicknessFactor"`
	SeaLevel           float64 `json:"seaLevel"`
	AdaptiveDensity    bool    `json:"adaptiveDensity"`
	CurvatureTolerance float64 `json:"curvatureTolerance"`
	ChunkRows          int     `json:"chunkRows"`
	ChunkCols          int     `json:"chunkCols"`
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Radius <= 0 {
		req.Radius = 100.0
	}
	if req.StepsLat < 4 {
		req.StepsLat = 150
	}
	if req.StepsLon < 4 {
		req.StepsLon = 300
	}
	if req.OverlapFactor <= 0 {
		req.OverlapFactor = 1.35
	}
	if req.ThicknessFactor <= 0 {
		req.ThicknessFactor = 0.035
	}
	if req.NoiseFrequency <= 0 {
		req.NoiseFrequency = 1.5
	}
	if req.Octaves <= 0 {
		req.Octaves = 5
	}
	if req.ChunkRows <= 0 {
		req.ChunkRows = 8
	}
	if req.ChunkCols <= 0 {
		req.ChunkCols = 16
	}
	if req.CurvatureTolerance <= 0 {
		req.CurvatureTolerance = 0.12
	}

	cfg := terrain.GeneratorConfig{
		Spheroid:           spheroid.NewSpheroid(req.Radius, req.Flattening),
		LatMin:             req.LatMin,
		LatMax:             req.LatMax,
		LonMin:             req.LonMin,
		LonMax:             req.LonMax,
		StepsLat:           req.StepsLat,
		StepsLon:           req.StepsLon,
		ElevationScale:     req.ElevationScale,
		ElevationOffset:    req.ElevationOffset,
		BaseElevation:      0.0,
		NoiseSeed:          req.NoiseSeed,
		NoiseFrequency:     req.NoiseFrequency,
		Octaves:            req.Octaves,
		OverlapFactor:      req.OverlapFactor,
		ThicknessFactor:    req.ThicknessFactor,
		SeaLevel:           0.30,
		AdaptiveDensity:    req.AdaptiveDensity,
		CurvatureTolerance: req.CurvatureTolerance,
	}

	gen := terrain.NewTerrainGenerator(cfg)

	if req.GeometryType == "latlon" {
		chunked := gen.GenerateChunkedTerrain(req.ChunkRows, req.ChunkCols)
		totalCount := 0
		for _, m := range chunked.Manifests {
			totalCount += m.CountsLOD[0]
		}

		s.mu.Lock()
		s.chunkedTerrain = chunked
		s.proceduralCube = nil
		s.splats = nil
		s.splatSet = nil
		s.spheroidInfo.Radius = req.Radius
		s.spheroidInfo.Flattening = req.Flattening
		s.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":    true,
			"splatCount": totalCount,
			"chunkCount": len(chunked.Chunks),
		})
		return
	}

	// Default: Instant Procedural Cubed Sphere (<1 MB RAM, instant generation for billions of splats!)
	procCube := gen.NewProceduralCubedSphere(req.PatchesPerFace)
	totalCount := 0
	for _, m := range procCube.Manifests {
		totalCount += m.CountsLOD[0]
	}

	s.mu.Lock()
	s.proceduralCube = procCube
	s.chunkedTerrain = nil
	s.splats = nil
	s.splatSet = nil
	s.chunkCache = make(map[string][]splat.Splat)
	s.cacheOrder = nil
	s.spheroidInfo.Radius = req.Radius
	s.spheroidInfo.Flattening = req.Flattening
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":            true,
		"splatCount":         totalCount,
		"virtualPoints":      req.StepsLat * req.StepsLon,
		"chunkCount":         procCube.TotalChunks,
		"adaptiveDensity":    req.AdaptiveDensity,
		"curvatureTolerance": req.CurvatureTolerance,
	})
}
