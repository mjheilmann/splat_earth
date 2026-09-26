package terrain

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// GEBCOTile stores an uncompressed in-memory 8-bit equirectangular quadrant DEM (10800x10800).
type GEBCOTile struct {
	Name   string
	Col    int     // 0 for [-180, -90], 1 for [-90, 0], 2 for [0, 90], 3 for [90, 180]
	Row    int     // 0 for [0, 90], 1 for [-90, 0]
	LatMin float64
	LatMax float64
	LonMin float64
	LonMax float64
	Width  int
	Height int
	Data   []byte
}

// GEBCOMosaic coordinates the 8 global 1km GEBCO quadrant tiles into a unified 43200x21600 DEM.
type GEBCOMosaic struct {
	Tiles       [4][2]*GEBCOTile // [col][row]
	GlobalWidth int              // 43200
	GlobalHeight int             // 21600
}

var (
	gebcoCache   *GEBCOMosaic
	gebcoCacheMu sync.Mutex
)

// FindDefaultGEBCOFiles searches standard paths for the 8 GEBCO 1km GeoTIFF quadrant tiles.
func FindDefaultGEBCOFiles() []string {
	searchDirs := []string{"images", ".", "../images", "..", "../../images", "../.."}
	for _, dir := range searchDirs {
		matches, err := filepath.Glob(filepath.Join(dir, "gebco_08_rev_elev_*_grey_geo.tif"))
		if err == nil && len(matches) >= 8 {
			return matches
		}
	}
	return nil
}

// LoadGEBCOMosaic loads and decodes the 8 GEBCO GeoTIFF quadrant tiles in parallel.
func LoadGEBCOMosaic(filePaths []string) (*GEBCOMosaic, error) {
	gebcoCacheMu.Lock()
	defer gebcoCacheMu.Unlock()

	if gebcoCache != nil {
		return gebcoCache, nil
	}

	mosaic := &GEBCOMosaic{
		GlobalWidth:  43200,
		GlobalHeight: 21600,
	}

	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error

	for _, path := range filePaths {
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
			continue
		}

		p := path
		c, r := col, row
		lMin, lMax, loMin, loMax := latMin, latMax, lonMin, lonMax

		wg.Add(1)
		go func() {
			defer wg.Done()
			tile, err := loadSingleGEBCOTile(p, c, r, lMin, lMax, loMin, loMax)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("failed to load GEBCO tile %s: %w", p, err)
				}
				errMu.Unlock()
				return
			}
			mosaic.Tiles[c][r] = tile
		}()
	}

	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	// Verify all 8 tiles were loaded
	loadedCount := 0
	for c := 0; c < 4; c++ {
		for r := 0; r < 2; r++ {
			if mosaic.Tiles[c][r] != nil {
				loadedCount++
			}
		}
	}

	if loadedCount < 8 {
		return nil, fmt.Errorf("only loaded %d of 8 required GEBCO tiles", loadedCount)
	}

	gebcoCache = mosaic
	return mosaic, nil
}

func loadSingleGEBCOTile(path string, col, row int, latMin, latMax, lonMin, lonMax float64) (*GEBCOTile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	order := binary.ByteOrder(binary.LittleEndian)
	if data[0] == 'M' && data[1] == 'M' {
		order = binary.BigEndian
	}
	ifdOffset := order.Uint32(data[4:8])
	numEntries := order.Uint16(data[ifdOffset : ifdOffset+2])

	var stripOffsets, stripByteCounts []uint32
	var width, height uint32

	cur := ifdOffset + 2
	for i := 0; i < int(numEntries); i++ {
		tag := order.Uint16(data[cur : cur+2])
		count := order.Uint32(data[cur+4 : cur+8])
		valOff := order.Uint32(data[cur+8 : cur+12])

		switch tag {
		case 256:
			width = valOff
		case 257:
			height = valOff
		case 273:
			stripOffsets = make([]uint32, count)
			for j := uint32(0); j < count; j++ {
				stripOffsets[j] = order.Uint32(data[valOff+j*4:])
			}
		case 279:
			stripByteCounts = make([]uint32, count)
			for j := uint32(0); j < count; j++ {
				stripByteCounts[j] = order.Uint32(data[valOff+j*4:])
			}
		}
		cur += 12
	}

	wInt := int(width)
	hInt := int(height)
	fullBuffer := make([]byte, wInt*hInt)

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	chunkSize := (hInt + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		yStart := w * chunkSize
		yEnd := yStart + chunkSize
		if yEnd > hInt {
			yEnd = hInt
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				off := stripOffsets[y]
				cnt := stripByteCounts[y]
				dst := fullBuffer[y*wInt : (y+1)*wInt]
				src := data[off : off+cnt]
				decodeTIFFLZW(dst, src)
			}
		}(yStart, yEnd)
	}
	wg.Wait()

	return &GEBCOTile{
		Name:   filepath.Base(path),
		Col:    col,
		Row:    row,
		LatMin: latMin,
		LatMax: latMax,
		LonMin: lonMin,
		LonMax: lonMax,
		Width:  wInt,
		Height: hInt,
		Data:   fullBuffer,
	}, nil
}

// Sample performs continuous bilinear sampling across the 43,200 x 21,600 global GEBCO mosaic.
// Returns normalized elevation and whether the sampled coordinate is terrestrial land (rawElev > 0).
// If isLand is false (water), it returns an underwater elevation below seaLevel.
func (m *GEBCOMosaic) Sample(seaLevel float64, lat, lon float64) (elev float64, isLand bool) {
	if m == nil {
		return seaLevel * 0.95, false
	}

	// Wrap longitude to [-180, 180)
	lon = math.Mod(lon+180.0, 360.0)
	if lon < 0 {
		lon += 360.0
	}
	lon -= 180.0

	// Clamp latitude to [-90, 90]
	if lat < -90.0 {
		lat = -90.0
	} else if lat > 90.0 {
		lat = 90.0
	}

	// Continuous global equirectangular coordinates (0 to 43200, 0 to 21600)
	u := (lon + 180.0) / 360.0 * float64(m.GlobalWidth)
	v := (90.0 - lat) / 180.0 * float64(m.GlobalHeight)

	x0 := int(math.Floor(u))
	y0 := int(math.Floor(v))

	fx := u - float64(x0)
	fy := v - float64(y0)

	x1 := x0 + 1
	y1 := y0 + 1

	// Global longitude wrap
	x0 = (x0%m.GlobalWidth + m.GlobalWidth) % m.GlobalWidth
	x1 = (x1%m.GlobalWidth + m.GlobalWidth) % m.GlobalWidth

	// Global latitude clamp
	if y0 < 0 {
		y0 = 0
	} else if y0 >= m.GlobalHeight {
		y0 = m.GlobalHeight - 1
	}
	if y1 < 0 {
		y1 = 0
	} else if y1 >= m.GlobalHeight {
		y1 = m.GlobalHeight - 1
	}

	val00 := m.getGlobalPixel(x0, y0)
	val10 := m.getGlobalPixel(x1, y0)
	val01 := m.getGlobalPixel(x0, y1)
	val11 := m.getGlobalPixel(x1, y1)

	top := val00*(1.0-fx) + val10*fx
	bot := val01*(1.0-fx) + val11*fx
	rawElev := top*(1.0-fy) + bot*fy

	if rawElev <= 0.0001 {
		// Ocean: return shelf depth below seaLevel
		return seaLevel * 0.98, false
	}

	// Terrestrial Land: map NASA Blue Marble [0.0, 1.0] (0m to 8848m) -> (seaLevel, 1.0]
	return seaLevel + rawElev*(1.0-seaLevel), true
}

func (m *GEBCOMosaic) getGlobalPixel(gx, gy int) float64 {
	col := gx / 10800
	if col >= 4 {
		col = 3
	}
	row := gy / 10800
	if row >= 2 {
		row = 1
	}

	tile := m.Tiles[col][row]
	if tile == nil || len(tile.Data) == 0 {
		return 0.5607
	}

	lx := gx % 10800
	ly := gy % 10800
	return float64(tile.Data[ly*10800+lx]) / 255.0
}

// decodeTIFFLZW decodes standard TIFF LZW compressed strips.
func decodeTIFFLZW(dst, src []byte) (int, error) {
	const (
		clearCode = 256
		eoiCode   = 257
	)

	prefix := make([]uint16, 4096)
	suffix := make([]byte, 4096)

	resetDict := func() int {
		for i := 0; i < 256; i++ {
			suffix[i] = byte(i)
		}
		return 258
	}

	nextCode := resetDict()
	codeSize := 9

	bitBuf := uint32(0)
	bitCount := 0
	srcIdx := 0
	dstIdx := 0
	oldCode := -1

	readCode := func() int {
		for bitCount < codeSize {
			if srcIdx >= len(src) {
				return -1
			}
			bitBuf = (bitBuf << 8) | uint32(src[srcIdx])
			srcIdx++
			bitCount += 8
		}
		code := int((bitBuf >> (bitCount - codeSize)) & ((1 << codeSize) - 1))
		bitCount -= codeSize
		return code
	}

	stack := make([]byte, 4096)

	for {
		code := readCode()
		if code < 0 || code == eoiCode {
			break
		}
		if code == clearCode {
			nextCode = resetDict()
			codeSize = 9
			oldCode = -1
			continue
		}

		cur := code
		stackIdx := 0

		if cur >= nextCode {
			if cur == nextCode && oldCode != -1 {
				stack[0] = stack[stackIdx]
				cur = oldCode
				stackIdx = 1
			} else {
				break
			}
		}

		for cur >= 256 {
			stack[stackIdx] = suffix[cur]
			stackIdx++
			cur = int(prefix[cur])
		}
		firstChar := suffix[cur]
		stack[stackIdx] = firstChar
		stackIdx++

		if code >= nextCode {
			stack[0] = firstChar
		}

		for s := stackIdx - 1; s >= 0; s-- {
			if dstIdx < len(dst) {
				dst[dstIdx] = stack[s]
				dstIdx++
			}
		}

		if oldCode != -1 && nextCode < 4096 {
			prefix[nextCode] = uint16(oldCode)
			suffix[nextCode] = firstChar
			nextCode++

			if nextCode == (1<<codeSize)-1 && codeSize < 12 {
				codeSize++
			}
		}
		oldCode = code
	}

	return dstIdx, nil
}
