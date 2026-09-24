package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mjheilmann/gaussian-splat-app/pkg/spheroid"
	"github.com/mjheilmann/gaussian-splat-app/pkg/splat"
	"github.com/mjheilmann/gaussian-splat-app/pkg/terrain"
)

func main() {
	defaultRaster := terrain.FindDefaultElevationRaster()

	outFile := flag.String("out", "splats/terrain.splat", "Output file path (.splat or .ply)")
	format := flag.String("format", "", "Output format: 'splat' or 'ply' (auto-detected from -out if omitted)")
	radius := flag.Float64("radius", 100.0, "Equatorial radius of the oblate spheroid (km)")
	flattening := flag.Float64("flattening", 0.05, "Flattening ratio f = (a - b) / a (e.g. 0.05 for 5% oblateness)")
	latMin := flag.Float64("lat-min", -90.0, "Minimum latitude in degrees (-90 to 90)")
	latMax := flag.Float64("lat-max", 90.0, "Maximum latitude in degrees (-90 to 90)")
	lonMin := flag.Float64("lon-min", -180.0, "Minimum longitude in degrees (-180 to 180)")
	lonMax := flag.Float64("lon-max", 180.0, "Maximum longitude in degrees (-180 to 180)")
	geometry := flag.String("geometry", "cubed-sphere", "Patch geometry: 'cubed-sphere' (recommended, no polar distortion) or 'latlon'")
	patches := flag.Int("patches", 0, "Patches per face axis for cubed-sphere (0 = Auto density-adaptive, or explicit e.g. 4 => 96 patches)")
	resLat := flag.Int("res-lat", 150, "Grid resolution along latitude (points)")
	resLon := flag.Int("res-lon", 300, "Grid resolution along longitude (points)")
	elevScale := flag.Float64("elev-scale", 12.0, "Max terrain elevation scale (km/units)")
	elevOffset := flag.Float64("elev-offset", 0.0, "Baseline elevation offset (-0.5 to +0.5)")
	overlap := flag.Float64("overlap", 1.35, "Splat size overlap factor for continuous coverage")
	thickness := flag.Float64("thickness", 0.035, "Splat normal thickness factor")
	rasterPath := flag.String("raster", defaultRaster, "Path to real-world DEM elevation raster image (e.g. images/World_elevation_map.png)")
	focusEnable := flag.Bool("focus-enable", false, "Enable steered high-density focus area (default 10 sq miles)")
	focusLat := flag.Float64("focus-lat", 0.0, "Focus area center latitude in degrees (-90 to 90)")
	focusLon := flag.Float64("focus-lon", 0.0, "Focus area center longitude in degrees (-180 to 180)")
	focusArea := flag.Float64("focus-area", 10.0, "Focus area size in square miles (default: 10.0 sq miles)")
	focusBoost := flag.Float64("focus-boost", 64.0, "Density boost multiplier in focus area (e.g. 16x - 128x)")
	flag.Parse()

	// Determine format
	outFormat := strings.ToLower(*format)
	if outFormat == "" {
		ext := strings.ToLower(filepath.Ext(*outFile))
		if ext == ".ply" {
			outFormat = "ply"
		} else {
			outFormat = "splat"
		}
	}

	fmt.Println("==================================================")
	fmt.Println("   Gaussian Splat Spheroid Terrain Generator      ")
	fmt.Println("==================================================")
	fmt.Printf("Geometry: %s (Patches=%d per face axis [0=Auto])\n", *geometry, *patches)
	fmt.Printf("Spheroid: Radius=%.1f km, Flattening=%.4f (Polar Radius=%.1f km)\n",
		*radius, *flattening, *radius*(1.0-*flattening))
	fmt.Printf("Bounds: Lat [%.1f, %.1f] deg, Lon [%.1f, %.1f] deg\n",
		*latMin, *latMax, *lonMin, *lonMax)
	fmt.Printf("Elevation: Scale=%.2f km, Offset=%.2f | Overlap=%.2f | Thickness=%.3f\n",
		*elevScale, *elevOffset, *overlap, *thickness)
	if *rasterPath != "" {
		if _, err := os.Stat(*rasterPath); err == nil {
			fmt.Printf("🗺️ DEM Raster: %s (Real-World Topography & Bathymetry)\n", *rasterPath)
		} else {
			fmt.Printf("⚠️ Raster specified at '%s' not found, using baseline spheroid\n", *rasterPath)
		}
	}
	geoTIFFs := terrain.FindDefaultGeoTIFFs()
	if len(geoTIFFs) > 0 {
		fmt.Printf("🏔️ High-Res LiDAR DEMs: Found %d GeoTIFF tiles (1-meter LiDAR Focus Layer)\n", len(geoTIFFs))
	}

	actualFocusLat := *focusLat
	actualFocusLon := *focusLon
	actualFocusArea := *focusArea
	actualFocusBoost := *focusBoost

	// Auto-default focus area to King County LiDAR bounds if focus-enable is set without custom lat/lon
	if *focusEnable && actualFocusLat == 0.0 && actualFocusLon == 0.0 && len(geoTIFFs) > 0 {
		actualFocusLat = 47.581
		actualFocusLon = -122.335
		actualFocusArea = 154.0 // ~400 km2 mosaic
	}

	if *focusEnable {
		fmt.Printf("🎯 Steered Focus Area: %.1f sq miles @ (Lat %.2f°, Lon %.2f°) with %.1fx Density Boost\n",
			actualFocusArea, actualFocusLat, actualFocusLon, actualFocusBoost)
	}
	fmt.Println("--------------------------------------------------")

	cfg := terrain.GeneratorConfig{
		Spheroid:        spheroid.NewSpheroid(*radius, *flattening),
		LatMin:          *latMin,
		LatMax:          *latMax,
		LonMin:          *lonMin,
		LonMax:          *lonMax,
		StepsLat:        *resLat,
		StepsLon:        *resLon,
		ElevationScale:  *elevScale,
		ElevationOffset: *elevOffset,
		BaseElevation:   0.0,
		OverlapFactor:   *overlap,
		ThicknessFactor: *thickness,
		SeaLevel:        143.0 / 255.0,
		RasterPath:      *rasterPath,
		GeoTIFFPaths:    geoTIFFs,
		FocusArea: terrain.DensityZone{
			Enabled:           *focusEnable,
			CenterLat:         actualFocusLat,
			CenterLon:         actualFocusLon,
			AreaSqMiles:       actualFocusArea,
			DensityMultiplier: actualFocusBoost,
			TransitionWidthKm: 0.0,
		},
	}

	startTime := time.Now()
	fmt.Print("Generating 3D Gaussian splat terrain from heightmap data...")
	gen := terrain.NewTerrainGenerator(cfg)
	var splats []splat.Splat
	if strings.ToLower(*geometry) == "latlon" {
		splats = gen.Generate()
	} else {
		splats = gen.GenerateCubedSphereSplats(*patches)
	}
	genDuration := time.Since(startTime)
	fmt.Printf(" Done (%v) - Generated %d splats\n", genDuration, len(splats))

	// Save file
	saveStart := time.Now()
	if parentDir := filepath.Dir(*outFile); parentDir != "" && parentDir != "." {
		_ = os.MkdirAll(parentDir, 0755)
	}
	fmt.Printf("Writing to %s (format: %s)...", *outFile, outFormat)
	var err error
	if outFormat == "ply" {
		err = splat.SavePLYFile(*outFile, splats)
	} else {
		err = splat.SaveSplatFile(*outFile, splats)
	}
	if err != nil {
		fmt.Printf("\nError saving file: %v\n", err)
		os.Exit(1)
	}
	saveDuration := time.Since(saveStart)

	fileInfo, statErr := os.Stat(*outFile)
	var fileSize int64
	if statErr == nil {
		fileSize = fileInfo.Size()
	}

	fmt.Printf(" Done (%v)\n", saveDuration)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Summary:\n")
	fmt.Printf(" - Splats Generated: %d\n", len(splats))
	fmt.Printf(" - File Size: %.2f MB (%d bytes)\n", float64(fileSize)/(1024*1024), fileSize)
	fmt.Printf(" - Total Time: %v\n", genDuration+saveDuration)
	fmt.Println("==================================================")
}
