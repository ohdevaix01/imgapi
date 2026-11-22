package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Stats struct {
	Requests  int64 `json:"requests"`
	CacheHits int64 `json:"cache_hits"`
	Errors    int64 `json:"errors"`
	Generated int64 `json:"generated"`
}

type Provider struct {
	Name    string
	Enabled bool
}

var (
	port      = env("PORT", "8080")
	dataDir   = env("DATA_DIR", "/data")
	cacheTTL  = time.Duration(envInt("CACHE_TTL", 604800)) * time.Second
	genTO     = time.Duration(envInt("GEN_TIMEOUT", 180)) * time.Second
	apiKey    = os.Getenv("API_KEY")
	hfToken   = os.Getenv("HF_TOKEN")
	pollKey   = os.Getenv("POLLINATIONS_TOKEN")
	order     = strings.Split(env("PROVIDER_ORDER", "flux,pollinations,horde"), ",")
	minGapMs  = envInt("POLLINATIONS_MIN_GAP_MS", 45000)

	mu        sync.RWMutex
	stats     Stats
	lastPoll  time.Time
	pollMu    sync.Mutex
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envInt(k string, d int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return d
}

// ---------- cache ----------

func cacheKey(prompt string, w, h int, seed int64, model string) string {
	raw := fmt.Sprintf("%s|%d|%d|%d|%s", strings.ToLower(strings.TrimSpace(prompt)), w, h, seed, model)
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:16])
}

func cachePath(key, ext string) string {
	return filepath.Join(dataDir, key+ext)
}

// findCached busca el archivo sin importar la extension.
func findCached(key string) (string, bool) {
	for _, ext := range []string{".jpg", ".png", ".webp"} {
		p := cachePath(key, ext)
		if st, err := os.Stat(p); err == nil {
			if cacheTTL == 0 || time.Since(st.ModTime()) < cacheTTL {
				return p, true
			}
			os.Remove(p)
		}
	}
	return "", false
}

func storeCached(key, ext string, data []byte) string {
	os.MkdirAll(dataDir, 0755)
	p := cachePath(key, ext)
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return ""
	}
	if err := os.Rename(tmp, p); err != nil {
		return ""
	}
	return p
}

// ---------- utilidades de imagen ----------

func sniffExt(data []byte) (string, string) {
	if len(data) < 12 {
		return "", ""
	}
	switch {
	case data[0] == 0xFF && data[1] == 0xD8:
		return ".jpg", "image/jpeg"
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		return ".png", "image/png"
	case bytes.HasPrefix(data, []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return ".webp", "image/webp"
	case bytes.HasPrefix(data, []byte("GIF8")):
		return ".gif", "image/gif"
	}
	return "", ""
}

// ---------- proveedores ----------

type genReq struct {
	Prompt   string
	Width    int
	Height   int
	Seed     int64
	Model    string
	Provider string
}

type genRes struct {
	Data     []byte
	Provider string
}

// flux usa el Space publico de HuggingFace por su API de Gradio (HTTP puro).
// Con HF_TOKEN la cuota de ZeroGPU es mucho mayor.
func genFlux(r genReq) (genRes, error) {
	space := env("FLUX_SPACE", "https://black-forest-labs-flux-1-schnell.hf.space")
	body, _ := json.Marshal(map[string]any{
		"data": []any{
			r.Prompt,
			float64(r.Seed),
			r.Seed == 0,
			float64(r.Width),
			float64(r.Height),
			4,
		},
		"session_hash": fmt.Sprintf("imgapi%d", time.Now().UnixNano()),
	})

	req, err := http.NewRequest("POST", space+"/gradio_api/call/infer", bytes.NewReader(body))
	if err != nil {
		return genRes{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return genRes{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return genRes{}, fmt.Errorf("flux call HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var started struct {
		EventID string `json:"event_id"`
	}
	if err := json.Unmarshal(raw, &started); err != nil || started.EventID == "" {
		return genRes{}, fmt.Errorf("flux sin event_id: %s", truncate(string(raw), 200))
	}

	sseReq, _ := http.NewRequest("GET", space+"/gradio_api/call/infer/"+started.EventID, nil)
	if hfToken != "" {
		sseReq.Header.Set("Authorization", "Bearer "+hfToken)
	}
	sseResp, err := (&http.Client{Timeout: genTO}).Do(sseReq)
	if err != nil {
		return genRes{}, err
	}
	defer sseResp.Body.Close()

	// El stream SSE manda "event: complete" y despues "data: [{url}, seed]".
	// La url apunta al archivo del Space y hay que descargarla aparte.
	scanner := bufio.NewScanner(sseResp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var event, dataLine string
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			dataLine = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
		if event == "error" {
			return genRes{}, fmt.Errorf("flux: error del Space: %s", truncate(dataLine, 200))
		}
		if event != "complete" || dataLine == "" {
			continue
		}
		var payload []struct {
			URL  string `json:"url"`
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(dataLine), &payload); err != nil || len(payload) == 0 {
			return genRes{}, fmt.Errorf("flux: payload inesperado: %s", truncate(dataLine, 200))
		}
		target := payload[0].URL
		if target == "" && payload[0].Path != "" {
			target = space + "/gradio_api/file=" + payload[0].Path
		}
		if target == "" {
			return genRes{}, fmt.Errorf("flux: sin url de imagen")
		}
		if strings.HasPrefix(target, "/") {
			target = space + target
		}
		imgReq, _ := http.NewRequest("GET", target, nil)
		if hfToken != "" {
			imgReq.Header.Set("Authorization", "Bearer "+hfToken)
		}
		imgResp, err := (&http.Client{Timeout: 90 * time.Second}).Do(imgReq)
		if err != nil {
			return genRes{}, err
		}
		data, _ := io.ReadAll(imgResp.Body)
		imgResp.Body.Close()
		if ext, _ := sniffExt(data); ext == "" {
			return genRes{}, fmt.Errorf("flux: la descarga no es imagen")
		}
		return genRes{Data: data, Provider: "flux"}, nil
	}
	return genRes{}, fmt.Errorf("flux: stream sin imagen utilizable")
}

// pollinations es un GET simple. Sin token: 1 imagen cada 45s por IP.
func genPollinations(r genReq) (genRes, error) {
	if minGapMs > 0 {
		pollMu.Lock()
		if !lastPoll.IsZero() {
			wait := time.Duration(minGapMs)*time.Millisecond - time.Since(lastPoll)
			if wait > 0 {
				pollMu.Unlock()
				time.Sleep(wait)
				pollMu.Lock()
			}
		}
		lastPoll = time.Now()
		pollMu.Unlock()
	}

	q := url.Values{}
	q.Set("width", strconv.Itoa(r.Width))
	q.Set("height", strconv.Itoa(r.Height))
	q.Set("nologo", "true")
	if r.Seed != 0 {
		q.Set("seed", strconv.FormatInt(r.Seed, 10))
	}
	if pollKey != "" {
		q.Set("token", pollKey)
	}
	if r.Model != "" {
		q.Set("model", r.Model)
	}

	endpoint := "https://image.pollinations.ai/prompt/" + url.PathEscape(r.Prompt) + "?" + q.Encode()

	// 503 y 502 son saturacion transitoria: vale reintentar antes de rendirse
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*6) * time.Second)
		}
		resp, err := (&http.Client{Timeout: genTO}).Get(endpoint)
		if err != nil {
			lastErr = err
			continue
		}
		data, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			lastErr = readErr
			continue
		}
		if resp.StatusCode == 503 || resp.StatusCode == 502 || resp.StatusCode == 429 {
			lastErr = fmt.Errorf("pollinations HTTP %d: %s", resp.StatusCode, truncate(string(data), 150))
			continue
		}
		if resp.StatusCode != 200 {
			return genRes{}, fmt.Errorf("pollinations HTTP %d: %s", resp.StatusCode, truncate(string(data), 200))
		}
		if ext, _ := sniffExt(data); ext == "" {
			lastErr = fmt.Errorf("pollinations no devolvio imagen: %s", truncate(string(data), 150))
			continue
		}
		return genRes{Data: data, Provider: "pollinations"}, nil
	}
	return genRes{}, lastErr
}

// horde es asincrono: se manda el job y se consulta hasta que termina.
func genHorde(r genReq) (genRes, error) {
	key := env("HORDE_API_KEY", "0000000000")
	// con la apikey anonima el presupuesto de kudos solo alcanza hasta 576x576
	w, h := r.Width, r.Height
	if key == "0000000000" {
		if w > 576 {
			w = 576
		}
		if h > 576 {
			h = 576
		}
	}
	payload, _ := json.Marshal(map[string]any{
		"prompt": r.Prompt,
		"params": map[string]any{
			"width":  w,
			"height": h,
			"steps":  25,
			"n":      1,
		},
		"nsfw": false,
	})

	req, _ := http.NewRequest("POST", "https://stablehorde.net/api/v2/generate/async", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("apikey", key)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return genRes{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 && resp.StatusCode != 202 {
		return genRes{}, fmt.Errorf("horde submit HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}
	var job struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(raw, &job); err != nil || job.ID == "" {
		return genRes{}, fmt.Errorf("horde sin id: %s", truncate(string(raw), 200))
	}

	deadline := time.Now().Add(genTO)
	for time.Now().Before(deadline) {
		time.Sleep(4 * time.Second)
		st, err := client.Get("https://stablehorde.net/api/v2/generate/status/" + job.ID)
		if err != nil {
			continue
		}
		raw, _ := io.ReadAll(st.Body)
		st.Body.Close()

		var done struct {
			Done        bool `json:"done"`
			Faulted     bool `json:"faulted"`
			Generations []struct {
				Img      string `json:"img"`
				Censored bool   `json:"censored"`
			} `json:"generations"`
		}
		if json.Unmarshal(raw, &done) != nil {
			continue
		}
		if done.Faulted {
			return genRes{}, fmt.Errorf("horde fallo el job")
		}
		if done.Done && len(done.Generations) > 0 {
			imgResp, err := client.Get(done.Generations[0].Img)
			if err != nil {
				return genRes{}, err
			}
			data, _ := io.ReadAll(imgResp.Body)
			imgResp.Body.Close()
			if ext, _ := sniffExt(data); ext == "" {
				return genRes{}, fmt.Errorf("horde devolvio algo que no es imagen")
			}
			return genRes{Data: data, Provider: "horde"}, nil
		}
	}
	return genRes{}, fmt.Errorf("horde: timeout esperando la imagen")
}

func validProvider(name string) bool {
	for _, p := range order {
		if strings.TrimSpace(p) == name {
			return true
		}
	}
	return false
}

func generate(r genReq, only string) (genRes, []string) {
	var failures []string
	for _, name := range order {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if only != "" && name != only {
			continue
		}
		var (
			res genRes
			err error
		)
		switch name {
		case "flux":
			res, err = genFlux(r)
		case "pollinations":
			res, err = genPollinations(r)
		case "horde":
			res, err = genHorde(r)
		default:
			continue
		}
		if err == nil {
			res.Provider = name
			return res, failures
		}
		log.Printf("proveedor %s fallo: %v", name, err)
		failures = append(failures, name+": "+err.Error())
	}
	return genRes{}, failures
}

// ---------- http ----------

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

func cors(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "X-API-Key, Content-Type")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	cors(w)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func authorized(r *http.Request) bool {
	if apiKey == "" {
		return true
	}
	return r.URL.Query().Get("key") == apiKey || r.Header.Get("X-API-Key") == apiKey
}

// parseReq acepta GET con query y POST con JSON.
func parseReq(r *http.Request) (genReq, error) {
	out := genReq{Width: 1024, Height: 1024}

	if r.Method == "POST" {
		var body struct {
			Prompt         string `json:"prompt"`
			Width          int    `json:"width"`
			Height         int    `json:"height"`
			Seed           int64  `json:"seed"`
			Model          string `json:"model"`
			Provider       string `json:"provider"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return out, fmt.Errorf("json invalido")
		}
		out.Prompt = body.Prompt
		if body.Width > 0 {
			out.Width = body.Width
		}
		if body.Height > 0 {
			out.Height = body.Height
		}
		out.Seed = body.Seed
		out.Model = body.Model
		out.Provider = body.Provider
	} else {
		q := r.URL.Query()
		out.Prompt = q.Get("prompt")
		if v := q.Get("width"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				out.Width = n
			}
		}
		if v := q.Get("height"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				out.Height = n
			}
		}
		if v := q.Get("seed"); v != "" {
			if n, err := strconv.ParseInt(v, 10, 64); err == nil {
				out.Seed = n
			}
		}
		out.Model = q.Get("model")
	}

	out.Prompt = strings.TrimSpace(out.Prompt)
	if out.Prompt == "" {
		return out, fmt.Errorf("falta el parametro prompt")
	}
	if len(out.Prompt) > 2000 {
		return out, fmt.Errorf("prompt demasiado largo (max 2000)")
	}
	out.Width = clamp(out.Width, 256, 2048)
	out.Height = clamp(out.Height, 256, 2048)
	// multiples de 8, que es lo que esperan los modelos
	out.Width = out.Width / 8 * 8
	out.Height = out.Height / 8 * 8
	return out, nil
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		cors(w)
		w.WriteHeader(204)
		return
	}
	if !authorized(r) {
		writeJSON(w, 401, map[string]string{"error": "api key invalida o ausente"})
		return
	}

	req, err := parseReq(r)
	if err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}

	// proveedor forzado, en orden de prioridad: ?provider= / X-Provider / body.provider
	only := r.URL.Query().Get("provider")
	if only == "" {
		only = r.Header.Get("X-Provider")
	}
	if only == "" {
		only = req.Provider
	}
	if only != "" && !validProvider(only) {
		writeJSON(w, 400, map[string]any{
			"error":     "proveedor desconocido",
			"providers": order,
		})
		return
	}

	wantJSON := r.URL.Query().Get("format") == "json"
	wantB64 := r.URL.Query().Get("b64") == "1"

	// el proveedor forzado entra en la clave: pedir horde no debe devolver
	// una imagen cacheada de flux
	key := cacheKey(req.Prompt, req.Width, req.Height, req.Seed, req.Model+"|"+only)
	start := time.Now()

	mu.Lock()
	stats.Requests++
	mu.Unlock()

	if p, ok := findCached(key); ok {
		mu.Lock()
		stats.CacheHits++
		mu.Unlock()
		serveFile(w, r, p, req, "cache", time.Since(start), wantJSON, wantB64)
		return
	}

	res, failures := generate(req, only)
	if res.Data == nil {
		mu.Lock()
		stats.Errors++
		mu.Unlock()
		writeJSON(w, 502, map[string]any{
			"error":     "todos los proveedores fallaron",
			"failures":  failures,
			"providers": order,
		})
		return
	}

	ext, mimeType := sniffExt(res.Data)
	_ = mimeType
	p := storeCached(key, ext, res.Data)

	mu.Lock()
	stats.Generated++
	mu.Unlock()

	if p != "" {
		serveFile(w, r, p, req, res.Provider, time.Since(start), wantJSON, wantB64)
		return
	}

	cors(w)
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("X-Provider", res.Provider)
	w.Write(res.Data)
}

func serveFile(w http.ResponseWriter, r *http.Request, path string, req genReq, provider string, took time.Duration, wantJSON, wantB64 bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		writeJSON(w, 500, map[string]string{"error": "no se pudo leer la imagen"})
		return
	}
	_, mimeType := sniffExt(data)
	if mimeType == "" {
		mimeType = mime.TypeByExtension(filepath.Ext(path))
	}

	if wantB64 || wantJSON {
		out := map[string]any{
			"prompt":   req.Prompt,
			"width":    req.Width,
			"height":   req.Height,
			"seed":     req.Seed,
			"provider": provider,
			"took_ms":  int(took.Milliseconds()),
			"mime":     mimeType,
		}
		if wantB64 {
			out["image_base64"] = base64.StdEncoding.EncodeToString(data)
		} else {
			out["image_url"] = "/image/" + strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		}
		writeJSON(w, 200, out)
		return
	}

	cors(w)
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("X-Provider", provider)
	w.Header().Set("X-Took-Ms", strconv.Itoa(int(took.Milliseconds())))
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Write(data)
}

func handleImage(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/image/")
	if id == "" || strings.ContainsAny(id, "/\\") {
		writeJSON(w, 400, map[string]string{"error": "id invalido"})
		return
	}
	p, ok := findCached(id)
	if !ok {
		writeJSON(w, 404, map[string]string{"error": "imagen no encontrada o expirada"})
		return
	}
	data, err := os.ReadFile(p)
	if err != nil {
		writeJSON(w, 500, map[string]string{"error": "no se pudo leer"})
		return
	}
	_, mimeType := sniffExt(data)
	cors(w)
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Write(data)
}

func main() {
	os.MkdirAll(dataDir, 0755)

	http.HandleFunc("/generate", handleGenerate)
	http.HandleFunc("/image/", handleImage)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"status": "ok",
			"providers": map[string]any{
				"order":             order,
				"flux_token":        hfToken != "",
				"pollinations_key":  pollKey != "",
			},
			"cache_ttl":  int(cacheTTL.Seconds()),
			"api_key_set": apiKey != "",
		})
	})
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		mu.RLock()
		s := stats
		mu.RUnlock()
		writeJSON(w, 200, s)
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			writeJSON(w, 404, map[string]any{
				"error": "ruta no encontrada",
				"paths": []string{"/generate", "/image/{id}", "/health", "/stats"},
			})
			return
		}
		http.ServeFile(w, r, "/app/public/index.html")
	})

	log.Printf("imgapi escuchando en :%s | proveedores: %v", port, order)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
