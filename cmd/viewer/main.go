package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/mjheilmann/gaussian-splat-app/pkg/viewer"
)

func main() {
	filePath := flag.String("file", "terrain.splat", "Path to .splat terrain file (if file does not exist, defaults are generated)")
	port := flag.Int("port", 8080, "HTTP server port for the 3D viewer")
	openBrowser := flag.Bool("open", true, "Automatically open the default browser")
	flag.Parse()

	// Check if file exists; if not, notify user that default spheroid terrain will be generated
	if _, err := os.Stat(*filePath); os.IsNotExist(err) {
		fmt.Printf("Note: Splat file '%s' not found. Starting with live procedural spheroid terrain.\n", *filePath)
		*filePath = ""
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
	} else {
		fmt.Println("Loaded: Live Procedural Oblate Spheroid Terrain")
	}
	fmt.Printf("Viewer URL: %s\n", url)
	fmt.Println("--------------------------------------------------")
	fmt.Println("Controls:")
	fmt.Println(" - Fly Mode: WASD (movement), Space/Shift (climb/dive), Mouse Drag (look)")
	fmt.Println(" - Orbit Mode: Left Mouse Drag (rotate), Right Drag (pan), Wheel (zoom)")
	fmt.Println(" - Tuning Panel: Scale Multiplier, Alpha Boost, Flight Speed, Biomes, Live GIS Generator")
	fmt.Println("==================================================")

	if *openBrowser {
		go func() {
			time.Sleep(500 * time.Millisecond)
			openURL(url)
		}()
	}

	if err := http.ListenAndServe(addr, server.Router()); err != nil {
		fmt.Printf("Server exited: %v\n", err)
	}
}

func openURL(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}
