package terrain

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"sync"
)

// ElevationRaster stores an in-memory 8-bit global equirectangular Digital Elevation Model (DEM).
type ElevationRaster struct {
	Width  int
	Height int
	Data   []byte // 1 byte per pixel: 0 (Mariana Trench) -> 143 (Sea Level) -> 255 (Mt. Everest)
}

var (
	rasterCache   *ElevationRaster
	rasterCacheMu sync.Mutex
	cachedPath    string
)

// LoadElevationRaster loads and converts an equirectangular image into an optimized grayscale DEM buffer.
// It caches the parsed raster in RAM to avoid redundant I/O and decoding across generator invocations.
func LoadElevationRaster(filePath string) (*ElevationRaster, error) {
	rasterCacheMu.Lock()
	defer rasterCacheMu.Unlock()

	if rasterCache != nil && cachedPath == filePath {
		return rasterCache, nil
	}

	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open elevation raster %s: %w", filePath, err)
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("failed to decode elevation raster %s: %w", filePath, err)
	}

	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	data := make([]byte, w*h)

	switch im := img.(type) {
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			rowOffset := (y + bounds.Min.Y) * im.Stride
			destOffset := y * w
			for x := 0; x < w; x++ {
				pxOffset := rowOffset + (x+bounds.Min.X)*4
				// R channel contains the 8-bit grayscale elevation
				data[destOffset+x] = im.Pix[pxOffset]
			}
		}
	case *image.RGBA:
		for y := 0; y < h; y++ {
			rowOffset := (y + bounds.Min.Y) * im.Stride
			destOffset := y * w
			for x := 0; x < w; x++ {
				pxOffset := rowOffset + (x+bounds.Min.X)*4
				data[destOffset+x] = im.Pix[pxOffset]
			}
		}
	case *image.Gray:
		for y := 0; y < h; y++ {
			rowOffset := (y + bounds.Min.Y) * im.Stride
			destOffset := y * w
			for x := 0; x < w; x++ {
				data[destOffset+x] = im.Pix[rowOffset+(x+bounds.Min.X)]
			}
		}
	default:
		for y := 0; y < h; y++ {
			destOffset := y * w
			for x := 0; x < w; x++ {
				r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
				// Standard luminance conversion
				lum := (r*299 + g*587 + b*114) / 1000
				data[destOffset+x] = byte(lum >> 8)
			}
		}
	}

	raster := &ElevationRaster{
		Width:  w,
		Height: h,
		Data:   data,
	}

	rasterCache = raster
	cachedPath = filePath

	return raster, nil
}

// Sample performs bilinear continuous sampling of the elevation raster at (lat, lon) in degrees.
// Returns normalized elevation in [0.0, 1.0].
func (r *ElevationRaster) Sample(lat, lon float64) float64 {
	if r == nil || len(r.Data) == 0 {
		return 0.56 // Default sea level baseline
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

	// Equirectangular mapping to continuous coordinates
	u := (lon + 180.0) / 360.0 * float64(r.Width)
	v := (90.0 - lat) / 180.0 * float64(r.Height)

	x0 := int(math.Floor(u))
	y0 := int(math.Floor(v))

	fx := u - float64(x0)
	fy := v - float64(y0)

	x1 := x0 + 1
	y1 := y0 + 1

	// Longitude wrap-around at 360° boundary
	x0 = (x0%r.Width + r.Width) % r.Width
	x1 = (x1%r.Width + r.Width) % r.Width

	// Latitude clamp at poles
	if y0 < 0 {
		y0 = 0
	} else if y0 >= r.Height {
		y0 = r.Height - 1
	}
	if y1 < 0 {
		y1 = 0
	} else if y1 >= r.Height {
		y1 = r.Height - 1
	}

	val00 := float64(r.Data[y0*r.Width+x0]) / 255.0
	val10 := float64(r.Data[y0*r.Width+x1]) / 255.0
	val01 := float64(r.Data[y1*r.Width+x0]) / 255.0
	val11 := float64(r.Data[y1*r.Width+x1]) / 255.0

	// Bilinear interpolation
	top := val00*(1.0-fx) + val10*fx
	bot := val01*(1.0-fx) + val11*fx
	return top*(1.0-fy) + bot*fy
}

// FindDefaultElevationRaster searches standard locations for the global DEM elevation raster map.
func FindDefaultElevationRaster() string {
	candidates := []string{
		"images/World_elevation_map.png",
		"World_elevation_map.png",
		"../../images/World_elevation_map.png",
		"../../World_elevation_map.png",
		"../images/World_elevation_map.png",
		"../World_elevation_map.png",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
