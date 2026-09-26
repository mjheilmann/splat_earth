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
	flattening := flag.Float64("flattening", spheroid.WGS84Flattening, "Flattening ratio f = (a - b) / a (0.0033528 for WGS-84)")
	latMin := flag.Float64("lat-min", -90.0, "Minimum latitude in degrees (-90 to 90)")
	latMax := flag.Float64("lat-max", 90.0, "Maximum latitude in degrees (-90 to 90)")
	lonMin := flag.Float64("lon-min", -180.0, "Minimum longitude in degrees (-180 to 180)")
	lonMax := flag.Float64("lon-max", 180.0, "Maximum longitude in degrees (-180 to 180)")
	geometry := flag.String("geometry", "cubed-sphere", "Patch geometry: 'cubed-sphere' (recommended, no polar distortion) or 'latlon'")
	patches := flag.Int("patches", 0, "Patches per face axis (0 = Auto optimal, e.g. 8 => 384 chunks)")
	resLat := flag.Int("res-lat", 0, "Grid resolution along latitude (0 = Auto native source height e.g. 10800)")
	resLon := flag.Int("res-lon", 0, "Grid resolution along longitude (0 = Auto native source width e.g. 21600)")
	elevScale := flag.Float64("elev-scale", 1.0, "Max terrain elevation scale in km (1.0 km for subtle relief, 0.311 km for true 1:1 physical scale)")
	elevOffset := flag.Float64("elev-offset", 0.0, "Baseline elevation offset (-0.5 to +0.5)")
	overlap := flag.Float64("overlap", 1.35, "Splat size overlap factor for continuous coverage")
	thickness := flag.Float64("thickness", 0.035, "Splat normal thickness factor")
	defaultBiome := terrain.FindDefaultBiomeConfig()
	rasterPath := flag.String("raster", defaultRaster, "Path to real-world DEM elevation raster image (e.g. images/World_elevation_map.png)")
	biomePath := flag.String("biome-config", defaultBiome, "Path to user-tunable JSON biome configuration file (e.g. config/biomes.json)")
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

	geoTIFFs := terrain.FindDefaultGeoTIFFs()
	gebcoFiles := terrain.FindDefaultGEBCOFiles()

	actualResLat := *resLat
	actualResLon := *resLon
	if actualResLat <= 0 || actualResLon <= 0 {
		if len(gebcoFiles) >= 8 {
			if actualResLon <= 0 {
				actualResLon = 43200
			}
			if actualResLat <= 0 {
				actualResLat = 21600
			}
		} else if *rasterPath != "" {
			if w, h, err := terrain.GetRasterDimensions(*rasterPath); err == nil && w > 0 && h > 0 {
				if actualResLon <= 0 {
					actualResLon = w
				}
				if actualResLat <= 0 {
					actualResLat = h
				}
			}
		}
		if actualResLat <= 0 {
			actualResLat = 150
		}
		if actualResLon <= 0 {
			actualResLon = 300
		}
	}

	actualPatches := *patches
	if actualPatches <= 0 {
		if actualResLon >= 4000 || len(gebcoFiles) > 0 || len(geoTIFFs) > 0 {
			actualPatches = 8 // 384 chunks for optimal high-res LOD partition
		}
	}

	fmt.Println("==================================================")
	fmt.Println("   Gaussian Splat Spheroid Terrain Generator      ")
	fmt.Println("==================================================")
	fmt.Printf("Geometry: %s (Patches=%d per face axis -> %d chunks)\n", *geometry, actualPatches, 6*actualPatches*actualPatches)
	fmt.Printf("Spheroid: Radius=%.1f km, Flattening=%.4f (Polar Radius=%.1f km)\n",
		*radius, *flattening, *radius*(1.0-*flattening))
	fmt.Printf("Bounds: Lat [%.1f, %.1f] deg, Lon [%.1f, %.1f] deg\n",
		*latMin, *latMax, *lonMin, *lonMax)
	fmt.Printf("Resolution: %d x %d (Auto native source limit)\n", actualResLon, actualResLat)
	fmt.Printf("Elevation: Scale=%.2f km, Offset=%.2f | Overlap=%.2f | Thickness=%.3f\n",
		*elevScale, *elevOffset, *overlap, *thickness)
	if len(gebcoFiles) >= 8 {
		fmt.Printf("🗺️ Global 1km GEBCO Mosaic: Found %d quadrant tiles (43,200 × 21,600 Native 1km Resolution)\n", len(gebcoFiles))
	} else if *rasterPath != "" {
		if _, err := os.Stat(*rasterPath); err == nil {
			fmt.Printf("🗺️ DEM Raster: %s (Real-World Topography & Bathymetry)\n", *rasterPath)
		} else {
			fmt.Printf("⚠️ Raster specified at '%s' not found, using baseline spheroid\n", *rasterPath)
		}
	}
	if len(geoTIFFs) > 0 {
		mosaic, _ := terrain.LoadGeoTIFFMosaic(geoTIFFs)
		if mosaic != nil {
			fmt.Printf("🏔️ High-Res LiDAR DEMs: Found %d GeoTIFF tiles (Auto-Detected: %.2f-meter Native Resolution)\n",
				len(geoTIFFs), mosaic.MinPixelScaleMeters())
		} else {
			fmt.Printf("🏔️ High-Res LiDAR DEMs: Found %d GeoTIFF tiles (1-meter LiDAR Focus Layer)\n", len(geoTIFFs))
		}
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
		StepsLat:        actualResLat,
		StepsLon:        actualResLon,
		ElevationScale:  *elevScale,
		ElevationOffset: *elevOffset,
		BaseElevation:   0.0,
		OverlapFactor:   *overlap,
		ThicknessFactor: *thickness,
		SeaLevel:        143.0 / 255.0,
		RasterPath:      *rasterPath,
		GEBCOPaths:      gebcoFiles,
		GeoTIFFPaths:    geoTIFFs,
		BiomeConfigPath: *biomePath,
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
	gen := terrain.NewTerrainGenerator(cfg)

	if parentDir := filepath.Dir(*outFile); parentDir != "" && parentDir != "." {
		_ = os.MkdirAll(parentDir, 0755)
	}

	if outFormat == "splat" && strings.ToLower(*geometry) != "latlon" {
		fmt.Printf("Generating compressed indexed multi-LOD splat container directly to %s...\n", *outFile)
		err := gen.WriteContainer(*outFile, actualPatches)
		if err != nil {
			fmt.Printf("\nError generating container: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Print("Generating 3D Gaussian splat terrain...")
		var splats []splat.Splat
		if strings.ToLower(*geometry) == "latlon" {
			splats = gen.Generate()
		} else {
			splats = gen.GenerateCubedSphereSplats(actualPatches)
		}
		genDuration := time.Since(startTime)
		fmt.Printf(" Done (%v) - Generated %d splats\n", genDuration, len(splats))

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
	}

	totalDuration := time.Since(startTime)
	fileInfo, statErr := os.Stat(*outFile)
	var fileSize int64
	if statErr == nil {
		fileSize = fileInfo.Size()
	}

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Summary:\n")
	fmt.Printf(" - Output File: %s\n", *outFile)
	fmt.Printf(" - File Size: %.2f MB (%d bytes)\n", float64(fileSize)/(1024*1024), fileSize)
	fmt.Printf(" - Compression: Enabled (ZLIB/DEFLATE Chunked LOD Streams)\n")
	fmt.Printf(" - Total Time: %v\n", totalDuration)
	fmt.Println("==================================================")
}
