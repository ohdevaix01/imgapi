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
	"math"
	"math/rand"
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
	port     = env("PORT", "8080")
	dataDir  = env("DATA_DIR", "/data")
	cacheTTL = time.Duration(envInt("CACHE_TTL", 604800)) * time.Second
	genTO    = time.Duration(envInt("GEN_TIMEOUT", 180)) * time.Second
	apiKey   = os.Getenv("API_KEY")
	hfToken  = os.Getenv("HF_TOKEN")
	pollKey  = os.Getenv("POLLINATIONS_TOKEN")
	order    = strings.Split(env("PROVIDER_ORDER", "flux,pollinations,horde"), ",")
	minGapMs = envInt("POLLINATIONS_MIN_GAP_MS", 45000)
	// techo de pixeles medido en la API nueva de pollinations; mas que esto
	// devuelve 422
	maxPixels = envInt("MAX_PIXELS", 2300000)

	// validar contra los proveedores que existen, no contra PROVIDER_ORDER:
	// si no, forzar uno que no esta en el orden da "proveedor desconocido"
	knownProviders = map[string]bool{"flux": true, "pollinations": true, "horde": true}

	mu       sync.RWMutex
	stats    Stats
	lastPoll time.Time
	pollMu   sync.Mutex
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
	// claves que puede mandar el cliente en cada llamado, pisan a las del .env
	HFToken    string
	PollToken  string
	HordeToken string
	// Cache en false saltea la cache: genera siempre una imagen nueva
	Cache bool
	// formato de salida, se pueden pedir por query o por body
	Format string
	B64    bool
}

// hfKey resuelve el token: el del request gana sobre el del entorno.
func (r genReq) hfKey() string {
	if r.HFToken != "" {
		return r.HFToken
	}
	return hfToken
}

func (r genReq) pollKeyValue() string {
	if r.PollToken != "" {
		return r.PollToken
	}
	return pollKey
}

func (r genReq) hordeKeyValue() string {
	if r.HordeToken != "" {
		return r.HordeToken
	}
	return env("HORDE_API_KEY", "0000000000")
}

type genRes struct {
	Data     []byte
	Provider string
}

// flux usa el Space publico de HuggingFace por su API de Gradio (HTTP puro).
// Con HF_TOKEN la cuota de ZeroGPU es mucho mayor.
func genFlux(r genReq) (genRes, error) {
	space := env("FLUX_SPACE", "https://black-forest-labs-flux-1-schnell.hf.space")
	session := fmt.Sprintf("imgapi%d", time.Now().UnixNano())
	body, _ := json.Marshal(map[string]any{
		"data": []any{
			r.Prompt,
			float64(r.Seed),
			r.Seed == 0,
			float64(r.Width),
			float64(r.Height),
			4,
		},
		"session_hash": session,
	})

	req, err := http.NewRequest("POST", space+"/gradio_api/call/infer", bytes.NewReader(body))
	if err != nil {
		return genRes{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if k := r.hfKey(); k != "" {
		req.Header.Set("Authorization", "Bearer "+k)
	}

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return genRes{}, err
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return genRes{}, fmt.Errorf("flux call HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var started struct {
		EventID string `json:"event_id"`
	}
	if err := json.Unmarshal(raw, &started); err != nil || started.EventID == "" {
		return genRes{}, fmt.Errorf("flux sin event_id: %s", truncate(string(raw), 200))
	}

	// gradio cambio el protocolo: el resultado ya no sale por
	// /call/<fn>/<event_id> sino por /queue/data?session_hash=..., con
	// msg=process_completed y la imagen adentro de output.data.
	sseReq, _ := http.NewRequest("GET", space+"/gradio_api/queue/data?session_hash="+session, nil)
	if k := r.hfKey(); k != "" {
		sseReq.Header.Set("Authorization", "Bearer "+k)
	}
	sseResp, err := (&http.Client{Timeout: genTO}).Do(sseReq)
	if err != nil {
		return genRes{}, err
	}
	defer sseResp.Body.Close()

	scanner := bufio.NewScanner(sseResp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		var msg struct {
			Msg     string `json:"msg"`
			EventID string `json:"event_id"`
			Success bool   `json:"success"`
			Output  struct {
				// array mixto: [ {file de la imagen}, seed(number) ]
				Data []json.RawMessage `json:"data"`
			} `json:"output"`
		}
		if err := json.Unmarshal([]byte(payload), &msg); err != nil {
			continue
		}
		// el stream es de toda la sesion; ignorar eventos de otro pedido
		if msg.EventID != "" && msg.EventID != started.EventID {
			continue
		}
		if msg.Msg == "process_generating" || msg.Msg == "estimation" ||
			msg.Msg == "process_starts" || msg.Msg == "progress" ||
			msg.Msg == "queue_full" || msg.Msg == "heartbeat" {
			continue
		}
		if msg.Msg == "close_stream" {
			break
		}
		if msg.Msg == "process_completed" {
			if !msg.Success {
				return genRes{}, fmt.Errorf("flux: el Space fallo al generar: %s", truncate(payload, 400))
			}
			if len(msg.Output.Data) == 0 {
				return genRes{}, fmt.Errorf("flux: respuesta sin imagen")
			}
			// el primer elemento es el archivo; los demas son escalares (seed)
			var file struct {
				URL  string `json:"url"`
				Path string `json:"path"`
			}
			if err := json.Unmarshal(msg.Output.Data[0], &file); err != nil || (file.URL == "" && file.Path == "") {
				return genRes{}, fmt.Errorf("flux: respuesta sin archivo: %s", truncate(string(msg.Output.Data[0]), 200))
			}
			target := file.URL
			if target == "" {
				target = space + "/gradio_api/file=" + file.Path
			}
			if target == "" {
				return genRes{}, fmt.Errorf("flux: sin url de imagen")
			}
			if strings.HasPrefix(target, "/") {
				target = space + target
			}
			imgReq, _ := http.NewRequest("GET", target, nil)
			if k := r.hfKey(); k != "" {
				imgReq.Header.Set("Authorization", "Bearer "+k)
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
	}
	if err := scanner.Err(); err != nil {
		return genRes{}, err
	}
	return genRes{}, fmt.Errorf("flux: stream sin imagen utilizable")
}

// pollinations tiene dos APIs: la nueva (gen.pollinations.ai) con Bearer y sin
// limite practico, y la clasica (image.pollinations.ai) sin key pero limitada a
// una imagen cada 45s por IP. Con key se usa la nueva.
func genPollinations(r genReq) (genRes, error) {
	key := r.pollKeyValue()

	q := url.Values{}
	q.Set("width", strconv.Itoa(r.Width))
	q.Set("height", strconv.Itoa(r.Height))
	if r.Seed != 0 {
		q.Set("seed", strconv.FormatInt(r.Seed, 10))
	}
	if r.Model != "" {
		q.Set("model", r.Model)
	}

	var endpoint, bearer string
	if key != "" {
		endpoint = "https://gen.pollinations.ai/image/" + url.PathEscape(r.Prompt) + "?" + q.Encode()
		bearer = key
	} else {
		// sin key hay que respetar el minuto entre llamados, o da 402
		q.Set("nologo", "true")
		endpoint = "https://image.pollinations.ai/prompt/" + url.PathEscape(r.Prompt) + "?" + q.Encode()
		if minGapMs > 0 {
			pollMu.Lock()
			if !lastPoll.IsZero() {
				if wait := time.Duration(minGapMs)*time.Millisecond - time.Since(lastPoll); wait > 0 {
					pollMu.Unlock()
					time.Sleep(wait)
					pollMu.Lock()
				}
			}
			lastPoll = time.Now()
			pollMu.Unlock()
		}
	}

	// 503 y 502 son saturacion transitoria: vale reintentar antes de rendirse
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*6) * time.Second)
		}
		req, _ := http.NewRequest("GET", endpoint, nil)
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		resp, err := (&http.Client{Timeout: genTO}).Do(req)
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
	key := r.hordeKeyValue()

	// sin kudos suficientes horde rechaza lo que pase de 576x576; en vez de
	// adivinar, se intenta y si lo rechaza se reenvia chico
	res, err := hordeJob(r, key, r.Width, r.Height)
	if err != nil && strings.Contains(err.Error(), "kudos") {
		w, h := min(576, r.Width), min(576, r.Height)
		if w != r.Width || h != r.Height {
			return hordeJob(r, key, w, h)
		}
	}
	return res, err
}

func hordeJob(r genReq, key string, w, h int) (genRes, error) {
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
	return knownProviders[name]
}

// generate recorre los proveedores. Si se fuerza uno, se usa ese solo, este o no
// en PROVIDER_ORDER: forzar flux no debe depender de que este en el orden.
func generate(r genReq, only string) (genRes, []string) {
	var failures []string

	candidates := order
	if only != "" {
		candidates = []string{only}
	}

	for _, name := range candidates {
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
			// el Space de HF tiene una cuota ZeroGPU chica que se agota seguido;
			// con key de pollinations ahi tambien hay FLUX y sin ese tope
			if err != nil && r.pollKeyValue() != "" {
				fb := r
				if fb.Model == "" {
					fb.Model = "black-forest-labs/flux.1-schnell"
				}
				var res2 genRes
				res2, err = genPollinations(fb)
				if err == nil {
					res2.Provider = "flux"
					res = res2
				}
			}
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
	out := genReq{Width: 1024, Height: 1024, Cache: true}

	if r.Method == "POST" {
		var body struct {
			Prompt     string `json:"prompt"`
			Width      int    `json:"width"`
			Height     int    `json:"height"`
			Seed       int64  `json:"seed"`
			Model      string `json:"model"`
			Provider   string `json:"provider"`
			HFToken    string `json:"hf_token"`
			PollToken  string `json:"pollinations_token"`
			HordeToken string `json:"horde_token"`
			Cache      *bool  `json:"cache"`
			Format     string `json:"format"`
			B64        bool   `json:"b64"`
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
		out.HFToken = body.HFToken
		out.PollToken = body.PollToken
		out.HordeToken = body.HordeToken
		if body.Cache != nil {
			out.Cache = *body.Cache
		}
		out.Format = body.Format
		out.B64 = body.B64
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
		out.Provider = q.Get("provider")
		out.HFToken = q.Get("hf_token")
		out.PollToken = q.Get("pollinations_token")
		out.HordeToken = q.Get("horde_token")
		// cache=0 desactiva tanto la lectura como la escritura
		if v := q.Get("cache"); v != "" {
			out.Cache = v != "0" && v != "false" && v != "no"
		}
		out.Format = q.Get("format")
		out.B64 = q.Get("b64") == "1"
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
	// horde exige multiplos de 64; redondear al mas cercano conserva mejor la
	// proporcion que truncar
	out.Width = round64(out.Width)
	out.Height = round64(out.Height)
	out.Width, out.Height = limitPixels(out.Width, out.Height, maxPixels)
	// con cache apagada, un seed aleatorio evita que el proveedor devuelva
	// siempre la misma imagen: pollinations cachea por prompt del lado suyo
	if !out.Cache && out.Seed == 0 {
		out.Seed = rand.Int63n(1<<31-1) + 1
	}
	return out, nil
}

func round64(v int) int {
	n := int(math.Round(float64(v)/64)) * 64
	return clamp(n, 256, 2048)
}

// limitPixels achica el par manteniendo la proporcion, para no pasar el techo
// de pixeles que acepta el proveedor.
func limitPixels(w, h, maxPx int) (int, int) {
	if w*h <= maxPx {
		return w, h
	}
	scale := math.Sqrt(float64(maxPx) / float64(w*h))
	w = round64(int(float64(w) * scale))
	h = round64(int(float64(h) * scale))
	for w*h > maxPx && (w > 256 || h > 256) {
		if w >= h && w > 256 {
			w -= 64
		} else if h > 256 {
			h -= 64
		} else {
			break
		}
	}
	return w, h
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
			"providers": []string{"flux", "pollinations", "horde"},
		})
		return
	}

	wantJSON := req.Format == "json"
	wantB64 := req.B64

	// el proveedor forzado entra en la clave: pedir horde no debe devolver
	// una imagen cacheada de flux
	key := cacheKey(req.Prompt, req.Width, req.Height, req.Seed, req.Model+"|"+only)
	start := time.Now()

	mu.Lock()
	stats.Requests++
	mu.Unlock()

	if req.Cache {
		if p, ok := findCached(key); ok {
			mu.Lock()
			stats.CacheHits++
			mu.Unlock()
			serveFile(w, r, p, req, "cache", time.Since(start), wantJSON, wantB64)
			return
		}
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
	p := ""
	if req.Cache {
		p = storeCached(key, ext, res.Data)
	}

	mu.Lock()
	stats.Generated++
	mu.Unlock()

	if p != "" {
		serveFile(w, r, p, req, res.Provider, time.Since(start), wantJSON, wantB64)
		return
	}
	// sin cache no hay archivo que servir: se responde con los bytes en mano
	serveBytes(w, r, res.Data, req, res.Provider, time.Since(start), wantJSON, wantB64, "")
}

func serveFile(w http.ResponseWriter, r *http.Request, path string, req genReq, provider string, took time.Duration, wantJSON, wantB64 bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		writeJSON(w, 500, map[string]string{"error": "no se pudo leer la imagen"})
		return
	}
	id := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	serveBytes(w, r, data, req, provider, took, wantJSON, wantB64, id)
}

// serveBytes arma la respuesta; con id vacio la imagen no quedo guardada.
func serveBytes(w http.ResponseWriter, r *http.Request, data []byte, req genReq, provider string, took time.Duration, wantJSON, wantB64 bool, id string) {
	_, mimeType := sniffExt(data)
	if mimeType == "" && id != "" {
		mimeType = mime.TypeByExtension(filepath.Ext(id))
	}
	if mimeType == "" {
		mimeType = "application/octet-stream"
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
		// sin id no hay url permanente, asi que va siempre el base64
		if wantB64 || id == "" {
			out["image_base64"] = base64.StdEncoding.EncodeToString(data)
		} else {
			out["image_url"] = "/image/" + id
		}
		writeJSON(w, 200, out)
		return
	}

	cors(w)
	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("X-Provider", provider)
	w.Header().Set("X-Took-Ms", strconv.Itoa(int(took.Milliseconds())))
	if req.Seed != 0 {
		w.Header().Set("X-Seed", strconv.FormatInt(req.Seed, 10))
	}
	// con cache apagada la respuesta no debe quedar en el navegador ni en
	// ningun proxy: si no, el mismo pedido devuelve la imagen vieja al instante
	if req.Cache {
		w.Header().Set("Cache-Control", "public, max-age=86400")
	} else {
		w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
	}
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
				"order":            order,
				"flux_token":       hfToken != "",
				"pollinations_key": pollKey != "",
			},
			"cache_ttl":   int(cacheTTL.Seconds()),
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
