package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/mjheilmann/gaussian-splat-app/pkg/terrain"
	"github.com/mjheilmann/gaussian-splat-app/pkg/viewer"
)

func main() {
	filePath := flag.String("file", "terrain.splat", "Path to .splat terrain file (if file does not exist, DEM data source is used)")
	port := flag.Int("port", 8080, "HTTP server port for the 3D viewer")
	openBrowser := flag.Bool("open", true, "Automatically open the default browser")
	flag.Parse()

	defaultRaster := terrain.FindDefaultElevationRaster()

	// Check if file exists in current path or under splats/
	if *filePath != "" {
		if _, err := os.Stat(*filePath); os.IsNotExist(err) {
			altPath := filepath.Join("splats", *filePath)
			if _, altErr := os.Stat(altPath); altErr == nil {
				*filePath = altPath
			} else {
				if defaultRaster != "" {
					fmt.Printf("Note: Splat file '%s' not found. Starting with live Real-World Earth DEM (%s).\n", *filePath, defaultRaster)
				} else {
					fmt.Printf("Note: Splat file '%s' not found. Starting with baseline spheroid terrain.\n", *filePath)
				}
				*filePath = ""
			}
		}
	}

	server, err := viewer.NewServer(*filePath)
	if err != nil {
		fmt.Printf("Error initializing viewer server: %v\n", err)
		os.Exit(1)
	}

	addr := fmt.Sprintf(":%d", *port)
	url := fmt.Sprintf("http://localhost:%d", *port)

	fmt.Println("==================================================")
	fmt.Println("       3D Gaussian Splat Spheroid Viewer          ")
	fmt.Println("==================================================")
	if *filePath != "" {
		fmt.Printf("Loaded File: %s\n", *filePath)
	} else if defaultRaster != "" {
		fmt.Printf("Mode: Live Real-World Earth DEM Streaming (%s)\n", defaultRaster)
	} else {
		fmt.Println("Mode: Live Oblate Spheroid Terrain")
	}
	fmt.Printf("Viewer URL: %s\n", url)
	fmt.Println("--------------------------------------------------")
	fmt.Println("Controls:")
	fmt.Println(" - Fly Mode: WASD (movement), Space/Shift (climb/dive), Mouse Drag (look)")
	fmt.Println(" - Orbit Mode: Left Mouse Drag (rotate), Right Drag (pan), Wheel (zoom)")
	fmt.Println(" - HUD & Drawer: Gaussian Splat Tuning, Focus Area Inspection, Lighting")
	fmt.Println("==================================================")

	if *openBrowser {
		go func() {
			time.Sleep(200 * time.Millisecond)
			openURL(url)
		}()
	}

	if err := http.ListenAndServe(addr, server.Router()); err != nil {
		fmt.Printf("Server failed: %v\n", err)
		os.Exit(1)
	}
}

func openURL(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}
