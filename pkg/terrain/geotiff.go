package terrain

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// GeoTIFFTile represents a single georeferenced Digital Elevation Model (DEM) tile in UTM coordinates.
type GeoTIFFTile struct {
	FilePath     string
	Width        int
	Height       int
	TileWidth    int
	TileHeight   int
	TilesAcross  int
	TilesDown    int
	EastingMin   float64
	EastingMax   float64
	NorthingMin  float64
	NorthingMax  float64
	PixelScaleX  float64
	PixelScaleY  float64
	LatMin       float64
	LatMax       float64
	LonMin       float64
	LonMax       float64
	UTMZone      int
	TileOffsets  []uint32
	TileCounts   []uint32
	tileCache    map[int][]float32
	tileCacheMu  sync.RWMutex
}

// GeoTIFFMosaic manages a collection of GeoTIFF DEM tiles for fast spatial elevation queries.
type GeoTIFFMosaic struct {
	Tiles   []*GeoTIFFTile
	LatMin  float64
	LatMax  float64
	LonMin  float64
	LonMax  float64
	mu      sync.RWMutex
}

// LoadGeoTIFFTile parses and prepares a GeoTIFF 32-bit floating point elevation DEM tile.
func LoadGeoTIFFTile(filePath string) (*GeoTIFFTile, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open geotiff %s: %w", filePath, err)
	}
	defer f.Close()

	var header [8]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, fmt.Errorf("failed reading tiff header %s: %w", filePath, err)
	}

	var bo binary.ByteOrder = binary.LittleEndian
	if header[0] == 'M' && header[1] == 'M' {
		bo = binary.BigEndian
	}

	magic := bo.Uint16(header[2:4])
	if magic != 42 {
		return nil, fmt.Errorf("unsupported tiff magic %d in %s", magic, filePath)
	}

	ifdOffset := int64(bo.Uint32(header[4:8]))
	if _, err := f.Seek(ifdOffset, 0); err != nil {
		return nil, err
	}

	var numEntries uint16
	if err := binary.Read(f, bo, &numEntries); err != nil {
		return nil, err
	}

	var width, height, tileWidth, tileHeight uint32
	var tileOffsets, tileCounts []uint32
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
		case 256: // ImageWidth
			width = valOffset
		case 257: // ImageLength
			height = valOffset
		case 322: // TileWidth
			tileWidth = valOffset
		case 323: // TileLength
			tileHeight = valOffset
		case 324: // TileOffsets
			f.Seek(int64(valOffset), 0)
			tileOffsets = make([]uint32, count)
			for k := 0; k < int(count); k++ {
				binary.Read(f, bo, &tileOffsets[k])
			}
		case 325: // TileByteCounts
			f.Seek(int64(valOffset), 0)
			tileCounts = make([]uint32, count)
			for k := 0; k < int(count); k++ {
				binary.Read(f, bo, &tileCounts[k])
			}
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
		return nil, fmt.Errorf("missing spatial tags in geotiff %s", filePath)
	}

	if tileWidth == 0 {
		tileWidth = 512
	}
	if tileHeight == 0 {
		tileHeight = 512
	}

	tilesAcross := int((width + tileWidth - 1) / tileWidth)
	tilesDown := int((height + tileHeight - 1) / tileHeight)

	sx := pixelScales[0]
	sy := pixelScales[1]
	originX := tiepoints[3]
	originY := tiepoints[4]
	maxX := originX + float64(width)*sx
	minY := originY - float64(height)*sy

	utmZone := 10 // Standard for WA King County
	latMin, lonMin := UTM10NToLatLon(originX, minY)
	latMax, lonMax := UTM10NToLatLon(maxX, originY)

	tile := &GeoTIFFTile{
		FilePath:    filePath,
		Width:       int(width),
		Height:      int(height),
		TileWidth:   int(tileWidth),
		TileHeight:  int(tileHeight),
		TilesAcross: tilesAcross,
		TilesDown:   tilesDown,
		EastingMin:  originX,
		EastingMax:  maxX,
		NorthingMin: minY,
		NorthingMax: originY,
		PixelScaleX: sx,
		PixelScaleY: sy,
		LatMin:      math.Min(latMin, latMax),
		LatMax:      math.Max(latMin, latMax),
		LonMin:      math.Min(lonMin, lonMax),
		LonMax:      math.Max(lonMin, lonMax),
		UTMZone:     utmZone,
		TileOffsets: tileOffsets,
		TileCounts:  tileCounts,
		tileCache:   make(map[int][]float32),
	}

	return tile, nil
}

// Sample evaluates elevation in meters at (lat, lon) with sub-pixel bilinear interpolation.
func (tile *GeoTIFFTile) Sample(lat, lon float64) (float64, bool) {
	if lat < tile.LatMin || lat > tile.LatMax || lon < tile.LonMin || lon > tile.LonMax {
		return 0, false
	}

	easting, northing := LatLonToUTM10N(lat, lon)
	if easting < tile.EastingMin || easting >= tile.EastingMax || northing <= tile.NorthingMin || northing > tile.NorthingMax {
		return 0, false
	}

	// Calculate continuous pixel coordinates
	u := (easting - tile.EastingMin) / tile.PixelScaleX
	v := (tile.NorthingMax - northing) / tile.PixelScaleY

	x0 := int(math.Floor(u))
	y0 := int(math.Floor(v))

	if x0 < 0 || x0 >= tile.Width-1 || y0 < 0 || y0 >= tile.Height-1 {
		return 0, false
	}

	fx := u - float64(x0)
	fy := v - float64(y0)

	h00 := tile.getPixel(x0, y0)
	h10 := tile.getPixel(x0+1, y0)
	h01 := tile.getPixel(x0, y0+1)
	h11 := tile.getPixel(x0+1, y0+1)

	// Ignore NoData / invalid values (e.g. < -500m)
	if h00 < -500.0 || h10 < -500.0 || h01 < -500.0 || h11 < -500.0 {
		return 0.0, true // Sea level / water body
	}

	top := float64(h00)*(1.0-fx) + float64(h10)*fx
	bot := float64(h01)*(1.0-fx) + float64(h11)*fx
	return top*(1.0-fy) + bot*fy, true
}

func (tile *GeoTIFFTile) getPixel(px, py int) float32 {
	tx := px / tile.TileWidth
	ty := py / tile.TileHeight
	tIdx := ty*tile.TilesAcross + tx

	if tIdx < 0 || tIdx >= len(tile.TileOffsets) {
		return 0
	}

	tile.tileCacheMu.RLock()
	data, found := tile.tileCache[tIdx]
	tile.tileCacheMu.RUnlock()

	if !found {
		var err error
		data, err = tile.loadTileData(tIdx)
		if err != nil {
			return 0
		}
		tile.tileCacheMu.Lock()
		tile.tileCache[tIdx] = data
		tile.tileCacheMu.Unlock()
	}

	localX := px % tile.TileWidth
	localY := py % tile.TileHeight
	return data[localY*tile.TileWidth+localX]
}

func (tile *GeoTIFFTile) loadTileData(tIdx int) ([]float32, error) {
	f, err := os.Open(tile.FilePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	offset := int64(tile.TileOffsets[tIdx])
	byteCount := int(tile.TileCounts[tIdx])
	if offset <= 0 || byteCount <= 0 {
		return make([]float32, tile.TileWidth*tile.TileHeight), nil
	}

	if _, err := f.Seek(offset, 0); err != nil {
		return nil, err
	}

	compressedData := make([]byte, byteCount)
	if _, err := io.ReadFull(f, compressedData); err != nil {
		return nil, err
	}

	decomp, err := decodeTiffLZW(compressedData, tile.TileWidth*tile.TileHeight*4)
	if err != nil {
		return nil, err
	}

	numPixels := tile.TileWidth * tile.TileHeight
	floats := make([]float32, numPixels)
	rowBytes := tile.TileWidth * 4

	for y := 0; y < tile.TileHeight; y++ {
		rowOffset := y * rowBytes
		if rowOffset+rowBytes > len(decomp) {
			break
		}
		// Apply Predictor 3 horizontal accumulation per byte plane
		for c := 1; c < tile.TileWidth; c++ {
			for b := 0; b < 4; b++ {
				idx := rowOffset + b*tile.TileWidth + c
				decomp[idx] += decomp[idx-1]
			}
		}
		for x := 0; x < tile.TileWidth; x++ {
			b0 := decomp[rowOffset+0*tile.TileWidth+x]
			b1 := decomp[rowOffset+1*tile.TileWidth+x]
			b2 := decomp[rowOffset+2*tile.TileWidth+x]
			b3 := decomp[rowOffset+3*tile.TileWidth+x]
			bits := uint32(b3) | (uint32(b2) << 8) | (uint32(b1) << 16) | (uint32(b0) << 24)
			floats[y*tile.TileWidth+x] = math.Float32frombits(bits)
		}
	}

	return floats, nil
}

// LoadGeoTIFFMosaic loads all matching GeoTIFF files into a seamless spatial mosaic.
func LoadGeoTIFFMosaic(filePaths []string) (*GeoTIFFMosaic, error) {
	mosaic := &GeoTIFFMosaic{
		LatMin: 90.0,
		LatMax: -90.0,
		LonMin: 180.0,
		LonMax: -180.0,
	}

	for _, p := range filePaths {
		tile, err := LoadGeoTIFFTile(p)
		if err != nil {
			continue
		}
		mosaic.Tiles = append(mosaic.Tiles, tile)
		if tile.LatMin < mosaic.LatMin {
			mosaic.LatMin = tile.LatMin
		}
		if tile.LatMax > mosaic.LatMax {
			mosaic.LatMax = tile.LatMax
		}
		if tile.LonMin < mosaic.LonMin {
			mosaic.LonMin = tile.LonMin
		}
		if tile.LonMax > mosaic.LonMax {
			mosaic.LonMax = tile.LonMax
		}
	}

	if len(mosaic.Tiles) == 0 {
		return nil, fmt.Errorf("no valid geotiff dem tiles found")
	}

	return mosaic, nil
}

// Sample queries the mosaic for elevation in meters at (lat, lon).
func (m *GeoTIFFMosaic) Sample(lat, lon float64) (float64, bool) {
	if m == nil || len(m.Tiles) == 0 {
		return 0, false
	}
	if lat < m.LatMin || lat > m.LatMax || lon < m.LonMin || lon > m.LonMax {
		return 0, false
	}

	for _, tile := range m.Tiles {
		if val, ok := tile.Sample(lat, lon); ok {
			return val, true
		}
	}
	return 0, false
}

// Contains returns true if the coordinate is within any tile's exact bounding box.
func (m *GeoTIFFMosaic) Contains(lat, lon float64) bool {
	if m == nil || len(m.Tiles) == 0 {
		return false
	}
	if lat < m.LatMin || lat > m.LatMax || lon < m.LonMin || lon > m.LonMax {
		return false
	}
	for _, tile := range m.Tiles {
		if lat >= tile.LatMin && lat <= tile.LatMax && lon >= tile.LonMin && lon <= tile.LonMax {
			return true
		}
	}
	return false
}

// Intersects returns true if the bounding box intersects any tile in the mosaic.
func (m *GeoTIFFMosaic) Intersects(latMin, latMax, lonMin, lonMax float64) bool {
	if m == nil || len(m.Tiles) == 0 {
		return false
	}
	if latMax < m.LatMin || latMin > m.LatMax || lonMax < m.LonMin || lonMin > m.LonMax {
		return false
	}
	for _, tile := range m.Tiles {
		if !(latMax < tile.LatMin || latMin > tile.LatMax || lonMax < tile.LonMin || lonMin > tile.LonMax) {
			return true
		}
	}
	return false
}

// MinPixelScaleMeters returns the finest horizontal resolution in meters among all tiles in the mosaic.
func (m *GeoTIFFMosaic) MinPixelScaleMeters() float64 {
	if m == nil || len(m.Tiles) == 0 {
		return 1.0
	}
	minScale := m.Tiles[0].PixelScaleX
	for _, t := range m.Tiles {
		if t.PixelScaleX > 0 && t.PixelScaleX < minScale {
			minScale = t.PixelScaleX
		}
		if t.PixelScaleY > 0 && t.PixelScaleY < minScale {
			minScale = t.PixelScaleY
		}
	}
	if minScale <= 0 {
		minScale = 1.0
	}
	return minScale
}

// FindDefaultGeoTIFFs finds any USGS 1-meter local LiDAR GeoTIFF files in images/ or workspace root.
func FindDefaultGeoTIFFs() []string {
	var paths []string
	patterns := []string{
		"images/USGS_1M_*.tif",
		"USGS_1M_*.tif",
		"images/*.tif",
		"*.tif",
		"../../images/USGS_1M_*.tif",
	}

	seen := make(map[string]bool)
	for _, pat := range patterns {
		matches, _ := filepath.Glob(pat)
		for _, m := range matches {
			base := strings.ToLower(filepath.Base(m))
			// Exclude global GEBCO quadrant tiles (managed separately by GEBCOMosaic)
			if strings.HasPrefix(base, "gebco_") {
				continue
			}
			if !seen[m] {
				seen[m] = true
				paths = append(paths, m)
			}
		}
	}
	return paths
}

// decodeTiffLZW performs standard TIFF LZW decompression (MSB-first, ClearCode 256, EOI 257).
func decodeTiffLZW(input []byte, expectedSize int) ([]byte, error) {
	out := make([]byte, 0, expectedSize)
	table := make([][]byte, 4096)
	for i := 0; i < 256; i++ {
		table[i] = []byte{byte(i)}
	}

	bitPos := 0
	codeSize := 9
	nextCode := 258

	readCode := func() (int, bool) {
		byteIdx := bitPos / 8
		bitOffset := bitPos % 8
		if byteIdx >= len(input) {
			return 0, false
		}
		var val uint32
		val = uint32(input[byteIdx]) << 24
		if byteIdx+1 < len(input) {
			val |= uint32(input[byteIdx+1]) << 16
		}
		if byteIdx+2 < len(input) {
			val |= uint32(input[byteIdx+2]) << 8
		}
		if byteIdx+3 < len(input) {
			val |= uint32(input[byteIdx+3])
		}
		code := int((val >> (32 - bitOffset - codeSize)) & ((1 << codeSize) - 1))
		bitPos += codeSize
		return code, true
	}

	var prevCode int = -1
	for {
		code, ok := readCode()
		if !ok || code == 257 { // EOI
			break
		}
		if code == 256 { // Clear code
			codeSize = 9
			nextCode = 258
			prevCode = -1
			continue
		}

		var entry []byte
		if code < nextCode && len(table[code]) > 0 {
			entry = table[code]
		} else if code == nextCode && prevCode != -1 {
			entry = append(append([]byte{}, table[prevCode]...), table[prevCode][0])
		} else {
			return out, fmt.Errorf("corrupt lzw code %d", code)
		}

		out = append(out, entry...)

		if prevCode != -1 && nextCode < 4096 {
			table[nextCode] = append(append([]byte{}, table[prevCode]...), entry[0])
			nextCode++
			if nextCode == (1<<codeSize)-1 && codeSize < 12 {
				codeSize++
			}
		}
		prevCode = code
	}

	return out, nil
}

// UTM10NToLatLon converts UTM Zone 10N (WGS84/NAD83) coordinates to Geodetic (Lat, Lon) in degrees.
func UTM10NToLatLon(easting, northing float64) (lat, lon float64) {
	a := 6378137.0
	f := 1.0 / 298.257223563
	e2 := 2.0*f - f*f
	ePrime2 := e2 / (1.0 - e2)
	k0 := 0.9996
	lon0 := -123.0 * math.Pi / 180.0

	x := easting - 500000.0
	y := northing

	M := y / k0
	mu := M / (a * (1.0 - e2/4.0 - 3.0*e2*e2/64.0 - 5.0*e2*e2*e2/256.0))

	e1 := (1.0 - math.Sqrt(1.0-e2)) / (1.0 + math.Sqrt(1.0-e2))
	phi1 := mu + (3.0*e1/2.0-27.0*e1*e1*e1/32.0)*math.Sin(2.0*mu) +
		(21.0*e1*e1/16.0-55.0*e1*e1*e1*e1/32.0)*math.Sin(4.0*mu) +
		(151.0*e1*e1*e1/96.0)*math.Sin(6.0*mu)

	sinPhi1 := math.Sin(phi1)
	cosPhi1 := math.Cos(phi1)
	tanPhi1 := math.Tan(phi1)

	N1 := a / math.Sqrt(1.0-e2*sinPhi1*sinPhi1)
	T1 := tanPhi1 * tanPhi1
	C1 := ePrime2 * cosPhi1 * cosPhi1
	R1 := a * (1.0 - e2) / math.Pow(1.0-e2*sinPhi1*sinPhi1, 1.5)
	D := x / (N1 * k0)

	latRad := phi1 - (N1*tanPhi1/R1)*(D*D/2.0-(5.0+3.0*T1+10.0*C1-4.0*C1*C1-9.0*ePrime2)*math.Pow(D, 4)/24.0+
		(61.0+90.0*T1+298.0*C1+45.0*T1*T1-252.0*ePrime2-3.0*C1*C1)*math.Pow(D, 6)/720.0)

	lonRad := lon0 + (D-(1.0+2.0*T1+C1)*math.Pow(D, 3)/6.0+
		(5.0-2.0*C1+28.0*T1-3.0*C1*C1+8.0*ePrime2+24.0*T1*T1)*math.Pow(D, 5)/120.0)/cosPhi1

	return latRad * 180.0 / math.Pi, lonRad * 180.0 / math.Pi
}

// LatLonToUTM10N converts Geodetic (Lat, Lon) in degrees to UTM Zone 10N Easting and Northing in meters.
func LatLonToUTM10N(lat, lon float64) (easting, northing float64) {
	a := 6378137.0
	f := 1.0 / 298.257223563
	e2 := 2.0*f - f*f
	ePrime2 := e2 / (1.0 - e2)
	k0 := 0.9996
	lon0 := -123.0 * math.Pi / 180.0

	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	tanLat := math.Tan(latRad)

	N := a / math.Sqrt(1.0-e2*sinLat*sinLat)
	T := tanLat * tanLat
	C := ePrime2 * cosLat * cosLat
	A := (lonRad - lon0) * cosLat

	M := a * ((1.0-e2/4.0-3.0*e2*e2/64.0-5.0*e2*e2*e2/256.0)*latRad -
		(3.0*e2/8.0+3.0*e2*e2/32.0+45.0*e2*e2*e2/1024.0)*math.Sin(2.0*latRad) +
		(15.0*e2*e2/256.0+45.0*e2*e2*e2/1024.0)*math.Sin(4.0*latRad) -
		(35.0*e2*e2*e2/3072.0)*math.Sin(6.0*latRad))

	easting = 500000.0 + k0*N*(A+(1.0-T+C)*math.Pow(A, 3)/6.0+(5.0-18.0*T+T*T+72.0*C-58.0*ePrime2)*math.Pow(A, 5)/120.0)
	northing = k0 * (M + N*tanLat*(A*A/2.0+(5.0-T+9.0*C+4.0*C*C)*math.Pow(A, 4)/24.0+(61.0-58.0*T+T*T+600.0*C-330.0*ePrime2)*math.Pow(A, 6)/720.0))

	return easting, northing
}
