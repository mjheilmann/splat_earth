package viewer

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestViewerServerEndpoints(t *testing.T) {
	server, err := NewServer("")
	if err != nil {
		t.Fatalf("Failed to create viewer server: %v", err)
	}

	handler := server.Router()

	// 1. Test /api/meta
	reqMeta := httptest.NewRequest("GET", "/api/meta", nil)
	recMeta := httptest.NewRecorder()
	handler.ServeHTTP(recMeta, reqMeta)

	if recMeta.Code != http.StatusOK {
		t.Fatalf("GET /api/meta returned status %d", recMeta.Code)
	}

	var meta map[string]interface{}
	if err := json.Unmarshal(recMeta.Body.Bytes(), &meta); err != nil {
		t.Fatalf("Failed to parse /api/meta response: %v", err)
	}
	if splatCount, ok := meta["splatCount"].(float64); !ok || splatCount <= 0 {
		t.Errorf("Expected positive splatCount in meta, got %v", meta["splatCount"])
	}

	// 2. Test /api/splat binary stream
	reqSplat := httptest.NewRequest("GET", "/api/splat", nil)
	recSplat := httptest.NewRecorder()
	handler.ServeHTTP(recSplat, reqSplat)

	if recSplat.Code != http.StatusOK {
		t.Fatalf("GET /api/splat returned status %d", recSplat.Code)
	}
	if recSplat.Body.Len() == 0 {
		t.Fatal("GET /api/splat returned empty binary stream")
	}

	// 3. Test /api/generate
	genPayload := map[string]interface{}{
		"radius":          100.0,
		"flattening":      0.05,
		"latMin":          -30.0,
		"latMax":          30.0,
		"lonMin":          -30.0,
		"lonMax":          30.0,
		"stepsLat":        20,
		"stepsLon":        20,
		"elevationScale":  5.0,
		"noiseSeed":       100,
		"noiseFrequency":  1.0,
		"octaves":         4,
		"overlapFactor":   1.2,
		"thicknessFactor": 0.2,
		"seaLevel":        0.3,
	}
	payloadBytes, _ := json.Marshal(genPayload)

	reqGen := httptest.NewRequest("POST", "/api/generate", bytes.NewReader(payloadBytes))
	recGen := httptest.NewRecorder()
	handler.ServeHTTP(recGen, reqGen)

	if recGen.Code != http.StatusOK {
		t.Fatalf("POST /api/generate returned status %d", recGen.Code)
	}

	var genRes map[string]interface{}
	if err := json.Unmarshal(recGen.Body.Bytes(), &genRes); err != nil {
		t.Fatalf("Failed to parse /api/generate response: %v", err)
	}
	if success, ok := genRes["success"].(bool); !ok || !success {
		t.Errorf("Expected success=true, got %v", genRes)
	}
	if splatCount, ok := genRes["splatCount"].(float64); !ok || splatCount <= 0 {
		t.Errorf("Expected positive splatCount, got %v", genRes["splatCount"])
	}
}
