package terrain

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// DatasetType indicates whether a dataset provides surface elevation or true color imagery.
type DatasetType int

const (
	DatasetTypeElevation DatasetType = iota
	DatasetTypeImagery
)

func (dt DatasetType) String() string {
	switch dt {
	case DatasetTypeElevation:
		return "Elevation DEM"
	case DatasetTypeImagery:
		return "Color Imagery"
	default:
		return "Unknown"
	}
}

// SurveyedImageInfo describes the detected format and georeferencing of a source image.
type SurveyedImageInfo struct {
	FilePath          string
	FormatDescription string
	Type              DatasetType
	Width             int
	Height            int
	LatMin            float64
	LatMax            float64
	LonMin            float64
	LonMax            float64
	PixelScaleMeters  float64
	PixelScaleDeg     float64
	IsGeoreferenced   bool
}

// ImageryDatasetTile stores decoded 24-bit RGB True Color imagery in memory for lock-free parallel sampling.
type ImageryDatasetTile struct {
	Info   SurveyedImageInfo
	Width  int
	Height int
	Data   []byte // Interleaved RGB pixels [y*Width*3 + x*3]
	loaded bool
	mu     sync.Mutex
}

func (t *ImageryDatasetTile) ensureLoaded() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.loaded {
		return nil
	}
	tile, err := openImageryTile(t.Info.FilePath, t.Info)
	if err != nil {
		return err
	}
	t.Width = tile.Width
	t.Height = tile.Height
	t.Data = tile.Data
	t.loaded = true
	return nil
}

// Sample evaluates RGB color at (lat, lon) on demand.
func (t *ImageryDatasetTile) Sample(lat, lon float64) ([3]uint8, bool) {
	if lat < t.Info.LatMin || lat > t.Info.LatMax || lon < t.Info.LonMin || lon > t.Info.LonMax {
		return [3]uint8{}, false
	}
	if !t.loaded {
		if err := t.ensureLoaded(); err != nil {
			return [3]uint8{}, false
		}
	}
	if t.Width == 0 || t.Height == 0 || len(t.Data) == 0 {
		return [3]uint8{}, false
	}
	u := (lon - t.Info.LonMin) / (t.Info.LonMax - t.Info.LonMin)
	v := (t.Info.LatMax - lat) / (t.Info.LatMax - t.Info.LatMin)

	x := int(u * float64(t.Width))
	y := int(v * float64(t.Height))

	if x < 0 {
		x = 0
	} else if x >= t.Width {
		x = t.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= t.Height {
		y = t.Height - 1
	}

	idx := (y*t.Width + x) * 3
	if idx+2 < len(t.Data) {
		return [3]uint8{t.Data[idx], t.Data[idx+1], t.Data[idx+2]}, true
	}
	return [3]uint8{}, false
}

// Close is a no-op for in-memory tiles.
func (t *ImageryDatasetTile) Close() {}

// ElevationSource is a common interface for querying elevation at (lat, lon).
type ElevationSource interface {
	SampleElevationMeters(lat, lon float64) (float64, bool)
	GetInfo() SurveyedImageInfo
}

// GeoTiffElevationSource wraps GeoTIFFTile (e.g. 1m USGS LiDAR).
type GeoTiffElevationSource struct {
	tile *GeoTIFFTile
	info SurveyedImageInfo
}

func (s *GeoTiffElevationSource) SampleElevationMeters(lat, lon float64) (float64, bool) {
	return s.tile.Sample(lat, lon)
}

func (s *GeoTiffElevationSource) GetInfo() SurveyedImageInfo {
	return s.info
}

// GEBCOElevationSource wraps GEBCOTile with lazy on-demand loading.
type GEBCOElevationSource struct {
	path   string
	tile   *GEBCOTile
	info   SurveyedImageInfo
	loaded bool
	mu     sync.Mutex
}

func (s *GEBCOElevationSource) ensureLoaded() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded {
		return nil
	}
	tile, err := loadSingleGEBCOTileWrapper(s.path)
	if err != nil {
		return err
	}
	s.tile = tile
	s.loaded = true
	return nil
}

func (s *GEBCOElevationSource) SampleElevationMeters(lat, lon float64) (float64, bool) {
	if lat < s.info.LatMin || lat > s.info.LatMax || lon < s.info.LonMin || lon > s.info.LonMax {
		return 0, false
	}
	if !s.loaded {
		if err := s.ensureLoaded(); err != nil {
			return 0, false
		}
	}
	if s.tile == nil {
		return 0, false
	}
	val, ok := s.tile.SampleFloat(lat, lon)
	if !ok {
		return 0, false
	}
	if strings.Contains(strings.ToLower(s.info.FilePath), "bath") || strings.Contains(strings.ToLower(s.tile.Name), "bath") {
		// GEBCO Bathymetry: 255 = 0m (sea level / coastline), < 255 = underwater depth down to -11000m
		if val < 255.0 {
			depthMeters := -((255.0 - val) / 255.0) * 11000.0
			return depthMeters, true
		}
		return 0.0, true
	}
	if val > 0.0 {
		// Terrestrial land elevation: [0, 255] -> [0m, 8848m]
		elevMeters := (val / 255.0) * 8848.0
		return elevMeters, true
	}
	// Ocean bed in elevation tile
	return -200.0, true
}

func (s *GEBCOElevationSource) GetInfo() SurveyedImageInfo {
	return s.info
}

// ImageCatalog surveys, catalogs, and manages all spatial source datasets.
type ImageCatalog struct {
	ElevationSources []ElevationSource
	ImageryTiles     []*ImageryDatasetTile
	UnusableImages   []string
	LatMin           float64
	LatMax           float64
	LonMin           float64
	LonMax           float64
}

var (
	surveyCatalogCache   *ImageCatalog
	surveyCatalogCacheMu sync.Mutex
)

// SurveyImageCatalog surveys all candidate directories for georeferenced datasets.
func SurveyImageCatalog(dirs ...string) *ImageCatalog {
	surveyCatalogCacheMu.Lock()
	defer surveyCatalogCacheMu.Unlock()

	if surveyCatalogCache != nil {
		return surveyCatalogCache
	}

	if len(dirs) == 0 {
		dirs = []string{"images", ".", "../images", "../../images"}
	}

	catalog := &ImageCatalog{
		LatMin: -90.0,
		LatMax: 90.0,
		LonMin: -180.0,
		LonMax: 180.0,
	}

	seenPaths := make(map[string]bool)
	var candidateFiles []string

	for _, dir := range dirs {
		if _, err := os.Stat(dir); err != nil {
			continue
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if ext == ".tif" || ext == ".tiff" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" {
				fullPath := filepath.Join(dir, e.Name())
				if !seenPaths[fullPath] {
					seenPaths[fullPath] = true
					candidateFiles = append(candidateFiles, fullPath)
				}
			}
		}
	}

	for _, path := range candidateFiles {
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".tif" && ext != ".tiff" {
			catalog.UnusableImages = append(catalog.UnusableImages, path)
			continue
		}

		info, err := inspectGeoTIFF(path)
		if err != nil || !info.IsGeoreferenced {
			catalog.UnusableImages = append(catalog.UnusableImages, path)
			continue
		}

		if info.Type == DatasetTypeImagery {
			catalog.ImageryTiles = append(catalog.ImageryTiles, &ImageryDatasetTile{
				Info: info,
			})
		} else if info.Type == DatasetTypeElevation {
			if strings.Contains(info.FormatDescription, "UTM") {
				tile, err := LoadGeoTIFFTile(path)
				if err == nil {
					catalog.ElevationSources = append(catalog.ElevationSources, &GeoTiffElevationSource{
						tile: tile,
						info: info,
					})
				} else {
					catalog.UnusableImages = append(catalog.UnusableImages, path)
				}
			} else {
				catalog.ElevationSources = append(catalog.ElevationSources, &GEBCOElevationSource{
					path: path,
					info: info,
				})
			}
		}
	}

	// Sort elevation sources by resolution ascending (finest resolution e.g. 1m first)
	sort.Slice(catalog.ElevationSources, func(i, j int) bool {
		return catalog.ElevationSources[i].GetInfo().PixelScaleMeters < catalog.ElevationSources[j].GetInfo().PixelScaleMeters
	})

	// Sort imagery tiles by resolution ascending
	sort.Slice(catalog.ImageryTiles, func(i, j int) bool {
		return catalog.ImageryTiles[i].Info.PixelScaleMeters < catalog.ImageryTiles[j].Info.PixelScaleMeters
	})

	surveyCatalogCache = catalog
	return catalog
}

// SampleElevationMeters evaluates physical elevation in meters (-11000m to +8848m) using the best available dataset.
func (cat *ImageCatalog) SampleElevationMeters(lat, lon float64) (float64, bool) {
	if cat == nil || len(cat.ElevationSources) == 0 {
		return -200.0, false
	}

	var bestBathSrc ElevationSource
	for _, src := range cat.ElevationSources {
		info := src.GetInfo()
		if lat >= info.LatMin && lat <= info.LatMax && lon >= info.LonMin && lon <= info.LonMax {
			if strings.Contains(strings.ToLower(info.FilePath), "bath") {
				if bestBathSrc == nil {
					bestBathSrc = src
				}
				continue
			}
			if elevMeters, ok := src.SampleElevationMeters(lat, lon); ok {
				if elevMeters > 0 {
					return elevMeters, true
				}
			}
		}
	}

	if bestBathSrc != nil {
		if depthMeters, ok := bestBathSrc.SampleElevationMeters(lat, lon); ok {
			return depthMeters, true
		}
	}

	return -200.0, true
}

// SampleElevation evaluates normalized elevation [0, 1] using the finest available DEM covering (lat, lon).
func (cat *ImageCatalog) SampleElevation(lat, lon, seaLevel float64) (float64, bool) {
	if cat == nil || len(cat.ElevationSources) == 0 {
		return seaLevel, false
	}

	eMeters, ok := cat.SampleElevationMeters(lat, lon)
	if !ok {
		return seaLevel, false
	}
	if eMeters >= 0 {
		hNorm := seaLevel + (eMeters/8848.0)*(1.0-seaLevel)
		return clampFloat(hNorm, 0.0, 1.0), true
	}
	hNorm := seaLevel + (eMeters/11000.0)*seaLevel
	return clampFloat(hNorm, 0.0, 1.0), true
}

// SampleElevationAndSlopes evaluates normalized elevation and metric slope gradients dh/dx, dh/dy on the model.
func (cat *ImageCatalog) SampleElevationAndSlopes(lat, lon, seaLevel, elevScale, semiMajorAxis float64) (elev, slopeE, slopeN, curv float64) {
	if cat == nil || len(cat.ElevationSources) == 0 {
		return seaLevel, 0, 0, 0
	}

	eCenter, okC := cat.SampleElevation(lat, lon, seaLevel)
	if !okC {
		return seaLevel, 0, 0, 0
	}

	bestScale := cat.BestPixelScaleMeters(lat, lon)
	var deltaDeg float64
	if bestScale <= 5.0 {
		deltaDeg = 0.000020 // ~2.0m for 1m LiDAR
	} else if bestScale <= 600.0 {
		deltaDeg = 0.004167 // ~463m for 500m data
	} else {
		deltaDeg = 0.008333 // ~925m for 1km GEBCO
	}

	deltaRad := deltaDeg * math.Pi / 180.0
	latRad := lat * math.Pi / 180.0
	cosLat := math.Max(0.01, math.Abs(math.Cos(latRad)))

	// 3x3 Sobel neighborhood for smooth, spike-free normal estimation on rugged mountain terrain
	eNW, _ := cat.SampleElevation(lat+deltaDeg, lon-deltaDeg, seaLevel)
	eN, _ := cat.SampleElevation(lat+deltaDeg, lon, seaLevel)
	eNE, _ := cat.SampleElevation(lat+deltaDeg, lon+deltaDeg, seaLevel)

	eW, _ := cat.SampleElevation(lat, lon-deltaDeg, seaLevel)
	eE, _ := cat.SampleElevation(lat, lon+deltaDeg, seaLevel)

	eSW, _ := cat.SampleElevation(lat-deltaDeg, lon-deltaDeg, seaLevel)
	eS, _ := cat.SampleElevation(lat-deltaDeg, lon, seaLevel)
	eSE, _ := cat.SampleElevation(lat-deltaDeg, lon+deltaDeg, seaLevel)

	// Sobel filtered height gradients (weighted 1-2-1 smoothing along orthogonal direction)
	dhE := ((eNE + 2.0*eE + eSE) - (eNW + 2.0*eW + eSW)) * (elevScale / 8.0)
	dhN := ((eNW + 2.0*eN + eNE) - (eSW + 2.0*eS + eSE)) * (elevScale / 8.0)

	dx := deltaRad * semiMajorAxis * cosLat
	dy := deltaRad * semiMajorAxis

	if dx > 0 {
		slopeE = dhE / dx
	}
	if dy > 0 {
		slopeN = dhN / dy
	}

	slopeE = clampFloat(slopeE, -2.5, 2.5)
	slopeN = clampFloat(slopeN, -2.5, 2.5)

	// Discrete 2D Laplacian curvature: negative = convex peak/ridge, positive = concave valley
	curv = ((eE + eW - 2.0*eCenter) + (eN + eS - 2.0*eCenter)) * 1000.0

	return eCenter, slopeE, slopeN, curv
}

// SampleImageryColor evaluates the finest RGB true color at (lat, lon).
func (cat *ImageCatalog) SampleImageryColor(lat, lon float64) ([3]uint8, bool) {
	if cat == nil || len(cat.ImageryTiles) == 0 {
		return [3]uint8{}, false
	}

	for _, tile := range cat.ImageryTiles {
		if lat >= tile.Info.LatMin && lat <= tile.Info.LatMax && lon >= tile.Info.LonMin && lon <= tile.Info.LonMax {
			if col, ok := tile.Sample(lat, lon); ok {
				return col, true
			}
		}
	}
	return [3]uint8{}, false
}

// BestPixelScaleMeters returns the finest horizontal resolution in meters covering (lat, lon).
func (cat *ImageCatalog) BestPixelScaleMeters(lat, lon float64) float64 {
	if cat == nil || len(cat.ElevationSources) == 0 {
		return 1000.0
	}
	for _, src := range cat.ElevationSources {
		info := src.GetInfo()
		if lat >= info.LatMin && lat <= info.LatMax && lon >= info.LonMin && lon <= info.LonMax {
			return info.PixelScaleMeters
		}
	}
	return 1000.0
}

func inspectGeoTIFF(path string) (SurveyedImageInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return SurveyedImageInfo{}, err
	}
	defer f.Close()

	var header [8]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return SurveyedImageInfo{}, err
	}

	var bo binary.ByteOrder = binary.LittleEndian
	if header[0] == 'M' && header[1] == 'M' {
		bo = binary.BigEndian
	}
	if bo.Uint16(header[2:4]) != 42 {
		return SurveyedImageInfo{}, fmt.Errorf("not a tiff")
	}

	ifdOffset := int64(bo.Uint32(header[4:8]))
	if _, err := f.Seek(ifdOffset, 0); err != nil {
		return SurveyedImageInfo{}, err
	}

	var numEntries uint16
	if err := binary.Read(f, bo, &numEntries); err != nil {
		return SurveyedImageInfo{}, err
	}

	var width, height uint32
	var bitsPerSample uint16
	var photometric uint16
	var samplesPerPixel uint16 = 1
	var pixelScales, tiepoints []float64

	for i := 0; i < int(numEntries); i++ {
		var tag, tagType uint16
		var count, valOffset uint32
		binary.Read(f, bo, &tag)
		binary.Read(f, bo, &tagType)
		binary.Read(f, bo, &count)
		binary.Read(f, bo, &valOffset)

		curPos, _ := f.Seek(0, 1)
		switch tag {
		case 256: // Width
			width = valOffset
		case 257: // Height
			height = valOffset
		case 258: // BitsPerSample
			bitsPerSample = uint16(valOffset)
		case 262: // PhotometricInterpretation
			photometric = uint16(valOffset)
		case 277: // SamplesPerPixel
			samplesPerPixel = uint16(valOffset)
		case 33550: // ModelPixelScaleTag
			f.Seek(int64(valOffset), 0)
			pixelScales = make([]float64, count)
			for k := 0; k < int(count); k++ {
				var d float64
				binary.Read(f, bo, &d)
				pixelScales[k] = d
			}
		case 33922: // ModelTiepointTag
			f.Seek(int64(valOffset), 0)
			tiepoints = make([]float64, count)
			for k := 0; k < int(count); k++ {
				var d float64
				binary.Read(f, bo, &d)
				tiepoints[k] = d
			}
		}
		f.Seek(curPos, 0)
	}

	if width == 0 || height == 0 || len(pixelScales) < 2 || len(tiepoints) < 6 {
		return SurveyedImageInfo{FilePath: path, IsGeoreferenced: false}, nil
	}

	sx := pixelScales[0]
	sy := pixelScales[1]
	originX := tiepoints[3]
	originY := tiepoints[4]

	info := SurveyedImageInfo{
		FilePath:        path,
		Width:           int(width),
		Height:          int(height),
		IsGeoreferenced: true,
	}

	// Detect if coordinate space is Projected UTM (meters) or Geodetic WGS-84 (degrees)
	if originX > 1000.0 || originX < -1000.0 || originY > 1000.0 || originY < -1000.0 {
		// UTM coordinates in meters
		maxX := originX + float64(width)*sx
		minY := originY - float64(height)*sy
		latMin, lonMin := UTM10NToLatLon(originX, minY)
		latMax, lonMax := UTM10NToLatLon(maxX, originY)

		info.LatMin = math.Min(latMin, latMax)
		info.LatMax = math.Max(latMin, latMax)
		info.LonMin = math.Min(lonMin, lonMax)
		info.LonMax = math.Max(lonMin, lonMax)
		info.PixelScaleMeters = sx
		info.PixelScaleDeg = (info.LonMax - info.LonMin) / float64(width)
		info.Type = DatasetTypeElevation
		info.FormatDescription = fmt.Sprintf("GeoTIFF (32-bit Float UTM %.1fm LiDAR)", sx)
	} else {
		// Geodetic coordinates in degrees
		maxLon := originX + float64(width)*sx
		minLat := originY - float64(height)*sy

		info.LatMin = math.Max(-90.0, math.Min(minLat, originY))
		info.LatMax = math.Min(90.0, math.Max(minLat, originY))
		info.LonMin = math.Max(-180.0, math.Min(originX, maxLon))
		info.LonMax = math.Min(180.0, math.Max(originX, maxLon))
		info.PixelScaleDeg = sx
		info.PixelScaleMeters = sx * 111319.5 // ~meters at equator

		if photometric == 2 || samplesPerPixel >= 3 || strings.Contains(strings.ToLower(path), "topo.bathy") {
			info.Type = DatasetTypeImagery
			info.FormatDescription = fmt.Sprintf("GeoTIFF (24-bit RGB True Color Imagery ~%.0fm)", info.PixelScaleMeters)
		} else {
			info.Type = DatasetTypeElevation
			if strings.Contains(strings.ToLower(path), "bath") {
				info.FormatDescription = fmt.Sprintf("GeoTIFF (%d-bit Grayscale Bathymetry ~%.0fm)", bitsPerSample, info.PixelScaleMeters)
			} else {
				info.FormatDescription = fmt.Sprintf("GeoTIFF (%d-bit Grayscale DEM ~%.0fm)", bitsPerSample, info.PixelScaleMeters)
			}
		}
	}

	return info, nil
}

func openImageryTile(path string, info SurveyedImageInfo) (*ImageryDatasetTile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [8]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, err
	}
	var bo binary.ByteOrder = binary.LittleEndian
	if header[0] == 'M' && header[1] == 'M' {
		bo = binary.BigEndian
	}
	ifdOffset := int64(bo.Uint32(header[4:8]))
	if _, err := f.Seek(ifdOffset, 0); err != nil {
		return nil, err
	}
	var numEntries uint16
	if err := binary.Read(f, bo, &numEntries); err != nil {
		return nil, err
	}

	var width, height uint32
	var stripOffsets, stripCounts []uint32

	for i := 0; i < int(numEntries); i++ {
		var tag, tagType uint16
		var count, valOffset uint32
		binary.Read(f, bo, &tag)
		binary.Read(f, bo, &tagType)
		binary.Read(f, bo, &count)
		binary.Read(f, bo, &valOffset)

		cur, _ := f.Seek(0, 1)
		switch tag {
		case 256:
			width = valOffset
		case 257:
			height = valOffset
		case 273:
			f.Seek(int64(valOffset), 0)
			stripOffsets = make([]uint32, count)
			for k := 0; k < int(count); k++ {
				binary.Read(f, bo, &stripOffsets[k])
			}
		case 279:
			f.Seek(int64(valOffset), 0)
			stripCounts = make([]uint32, count)
			for k := 0; k < int(count); k++ {
				binary.Read(f, bo, &stripCounts[k])
			}
		}
		f.Seek(cur, 0)
	}

	wInt := int(width)
	hInt := int(height)
	if wInt == 0 || hInt == 0 || len(stripOffsets) == 0 {
		return nil, fmt.Errorf("missing spatial strips in %s", path)
	}

	// 2:1 downsampling to 10800x10800 RGB (~333 MB per tile in RAM, 43200x21600 global resolution)
	scale := 2
	outW := wInt / scale
	outH := hInt / scale
	outBuf := make([]byte, outW*outH*3)

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	chunk := (outH + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		yStart := w * chunk
		yEnd := yStart + chunk
		if yEnd > outH {
			yEnd = outH
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for outY := y0; outY < y1; outY++ {
				srcY := outY * scale
				if srcY >= len(stripOffsets) || srcY >= len(stripCounts) {
					continue
				}
				offset := int64(stripOffsets[srcY])
				count := int(stripCounts[srcY])
				buf := make([]byte, count)
				if _, err := f.ReadAt(buf, offset); err == nil {
					zr, err := zlib.NewReader(bytes.NewReader(buf))
					if err == nil {
						decomp, _ := io.ReadAll(zr)
						zr.Close()
						if len(decomp) >= wInt*3 {
							dstRow := outY * outW * 3
							for outX := 0; outX < outW; outX++ {
								srcX := outX * scale * 3
								outBuf[dstRow+outX*3] = decomp[srcX]
								outBuf[dstRow+outX*3+1] = decomp[srcX+1]
								outBuf[dstRow+outX*3+2] = decomp[srcX+2]
							}
						}
					}
				}
			}
		}(yStart, yEnd)
	}
	wg.Wait()

	return &ImageryDatasetTile{
		Info:   info,
		Width:  outW,
		Height: outH,
		Data:   outBuf,
	}, nil
}

func loadSingleGEBCOTileWrapper(path string) (*GEBCOTile, error) {
	base := strings.ToLower(filepath.Base(path))
	var col, row int
	var latMin, latMax, lonMin, lonMax float64

	if strings.Contains(base, "_a1_") {
		col, row = 0, 0
		lonMin, lonMax = -180.0, -90.0
		latMin, latMax = 0.0, 90.0
	} else if strings.Contains(base, "_a2_") {
		col, row = 0, 1
		lonMin, lonMax = -180.0, -90.0
		latMin, latMax = -90.0, 0.0
	} else if strings.Contains(base, "_b1_") {
		col, row = 1, 0
		lonMin, lonMax = -90.0, 0.0
		latMin, latMax = 0.0, 90.0
	} else if strings.Contains(base, "_b2_") {
		col, row = 1, 1
		lonMin, lonMax = -90.0, 0.0
		latMin, latMax = -90.0, 0.0
	} else if strings.Contains(base, "_c1_") {
		col, row = 2, 0
		lonMin, lonMax = 0.0, 90.0
		latMin, latMax = 0.0, 90.0
	} else if strings.Contains(base, "_c2_") {
		col, row = 2, 1
		lonMin, lonMax = 0.0, 90.0
		latMin, latMax = -90.0, 0.0
	} else if strings.Contains(base, "_d1_") {
		col, row = 3, 0
		lonMin, lonMax = 90.0, 180.0
		latMin, latMax = 0.0, 90.0
	} else if strings.Contains(base, "_d2_") {
		col, row = 3, 1
		lonMin, lonMax = 90.0, 180.0
		latMin, latMax = -90.0, 0.0
	} else {
		return nil, fmt.Errorf("unrecognized gebco tile format %s", path)
	}

	return LoadSingleGEBCOTile(path, col, row, latMin, latMax, lonMin, lonMax)
}
