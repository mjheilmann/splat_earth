package viewer

import (
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

	// 2. Test /api/chunks
	reqChunks := httptest.NewRequest("GET", "/api/chunks", nil)
	recChunks := httptest.NewRecorder()
	handler.ServeHTTP(recChunks, reqChunks)

	if recChunks.Code != http.StatusOK {
		t.Fatalf("GET /api/chunks returned status %d", recChunks.Code)
	}

	var chunksRes map[string]interface{}
	if err := json.Unmarshal(recChunks.Body.Bytes(), &chunksRes); err != nil {
		t.Fatalf("Failed to parse /api/chunks response: %v", err)
	}
	chunksList, ok := chunksRes["chunks"].([]interface{})
	if !ok || len(chunksList) == 0 {
		t.Fatalf("Expected non-empty chunks list, got %v", chunksRes)
	}

	// 3. Test /api/chunk binary stream for chunk 0, LOD 0
	reqChunkBin := httptest.NewRequest("GET", "/api/chunk?id=0&lod=0", nil)
	recChunkBin := httptest.NewRecorder()
	handler.ServeHTTP(recChunkBin, reqChunkBin)

	if recChunkBin.Code != http.StatusOK {
		t.Fatalf("GET /api/chunk returned status %d", recChunkBin.Code)
	}
	if recChunkBin.Body.Len() == 0 {
		t.Fatalf("Expected binary chunk splat data, got 0 bytes")
	}

	// 4. Test /api/splat binary stream
	reqSplat := httptest.NewRequest("GET", "/api/splat", nil)
	recSplat := httptest.NewRecorder()
	handler.ServeHTTP(recSplat, reqSplat)

	if recSplat.Code != http.StatusOK {
		t.Fatalf("GET /api/splat returned status %d", recSplat.Code)
	}
	if recSplat.Body.Len() == 0 {
		t.Fatal("GET /api/splat returned empty binary stream")
	}
}
