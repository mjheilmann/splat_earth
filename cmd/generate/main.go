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
	outFile := flag.String("out", "terrain.splat", "Output file path (.splat or .ply)")
	format := flag.String("format", "", "Output format: 'splat' or 'ply' (auto-detected from -out if omitted)")
	radius := flag.Float64("radius", 100.0, "Equatorial radius of the oblate spheroid")
	flattening := flag.Float64("flattening", 0.05, "Flattening ratio f = (a - b) / a (e.g. 0.05 for 5% oblateness)")
	latMin := flag.Float64("lat-min", -90.0, "Minimum latitude in degrees (-90 to 90)")
	latMax := flag.Float64("lat-max", 90.0, "Maximum latitude in degrees (-90 to 90)")
	lonMin := flag.Float64("lon-min", -180.0, "Minimum longitude in degrees (-180 to 180)")
	lonMax := flag.Float64("lon-max", 180.0, "Maximum longitude in degrees (-180 to 180)")
	geometry := flag.String("geometry", "cubed-sphere", "Patch geometry: 'cubed-sphere' (recommended, no polar distortion) or 'latlon'")
	patches := flag.Int("patches", 0, "Patches per face axis for cubed-sphere (0 = Auto density-adaptive, or explicit e.g. 4 => 96 patches)")
	resLat := flag.Int("res-lat", 150, "Grid resolution along latitude (points)")
	resLon := flag.Int("res-lon", 300, "Grid resolution along longitude (points)")
	elevScale := flag.Float64("elev-scale", 12.0, "Max procedural terrain elevation scale (km/units)")
	elevOffset := flag.Float64("elev-offset", 0.0, "Baseline elevation offset (-0.5 to +0.5 to shift continent/sea ratio)")
	freq := flag.Float64("freq", 1.8, "Procedural noise frequency")
	octaves := flag.Int("octaves", 6, "Noise fractal octaves")
	seed := flag.Int64("seed", 42, "Seed for terrain generator")
	overlap := flag.Float64("overlap", 1.35, "Splat size overlap factor for continuous coverage")
	thickness := flag.Float64("thickness", 0.15, "Splat normal thickness factor")
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
	fmt.Printf("Elevation: Scale=%.2f km, Offset=%.2f | Overlap=%.2f | Thickness=%.2f\n",
		*elevScale, *elevOffset, *overlap, *thickness)
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
		NoiseSeed:       *seed,
		NoiseFrequency:  *freq,
		Octaves:         *octaves,
		OverlapFactor:   *overlap,
		ThicknessFactor: *thickness,
		SeaLevel:        0.30,
	}

	startTime := time.Now()
	fmt.Print("Generating 3D Gaussian splat terrain...")
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
	fmt.Printf("Writing to %s (format: %s)...", *outFile, outFormat)
	var err error
	if outFormat == "ply" {
		err = splat.SavePLYFile(*outFile, splats)
	} else {
		err = splat.SaveSplatFile(*outFile, splats)
	}

	if err != nil {
		fmt.Printf("\nError saving splat file: %v\n", err)
		os.Exit(1)
	}
	saveDuration := time.Since(saveStart)

	fileInfo, err := os.Stat(*outFile)
	fileSizeBytes := int64(0)
	if err == nil {
		fileSizeBytes = fileInfo.Size()
	}

	fmt.Printf(" Done (%v)\n", saveDuration)
	fmt.Printf("Output File: %s (%.2f MB)\n", *outFile, float64(fileSizeBytes)/(1024*1024))
	fmt.Println("==================================================")
}
