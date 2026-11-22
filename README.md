# imgapi

Generador de imagenes con IA detras de una API propia escrita en Go. Misma idea que el
stack de SearXNG: una sola puerta de entrada, cache en disco y varios proveedores atras
que se reemplazan solos si uno falla.

Corre en un contenedor de **8.4 MB** sobre `scratch`, sin sistema operativo, y ocupa
unos pocos MB de RAM. No necesita GPU propia: usa proveedores externos gratuitos.

## Proveedores

| Proveedor | Token | Limite sin token | Velocidad tipica |
|---|---|---|---|
| `flux` | HF_TOKEN (gratis) | cuota ZeroGPU de ~5 min por dia | 10 s |
| `pollinations` | POLLINATIONS_TOKEN (gratis) | 1 imagen cada 45 s por IP | 4 s |
| `horde` | HORDE_API_KEY (gratis) | tope de 576x576 sin kudos | 30 s |

El orden de intento se configura con `PROVIDER_ORDER`. Si el primero falla, pasa al
siguiente sin que el cliente se entere. Forzar `?provider=` no depende de esa lista:
funciona aunque el proveedor no este en el orden.

Con `POLLINATIONS_TOKEN` el servicio usa la API nueva (`gen.pollinations.ai`), que no
tiene el limite de 45 s y ademas sirve los modelos de FLUX. Por eso, si `flux` falla por
cuota de ZeroGPU y hay key de Pollinations, la imagen se genera igual: sale por
Pollinations con el modelo `black-forest-labs/flux.1-schnell` y la respuesta sigue
diciendo `flux`.

En `horde`, una cuenta recien creada tiene 0 kudos: el servicio intenta el tamano pedido
y, si el servidor lo rechaza por falta de kudos, reenvia el trabajo a 576x576.

## Uso rapido

```bash
cp .env.example .env
docker compose up -d --build
```

Despues:

```bash
# la imagen directa
curl "http://localhost:8080/generate?prompt=un+gato+negro" -o gato.jpg

# forzando un proveedor
curl "http://localhost:8080/generate?prompt=un+gato+negro&provider=horde" -o gato.jpg

# JSON con la url
curl "http://localhost:8080/generate?prompt=un+gato+negro&format=json"

# JSON con base64
curl "http://localhost:8080/generate?prompt=un+gato+negro&b64=1"
```

## Parametros

| Parametro | Defecto | Que hace |
|---|---|---|
| `prompt` | — | obligatorio, maximo 2000 caracteres |
| `width` / `height` | 1024 | entre 256 y 2048, se redondea al multiplo de 64 mas cercano y se achica si pasa el techo de pixeles |
| `seed` | 0 | con el mismo seed repite la misma imagen |
| `model` | — | pasa el modelo al proveedor |
| `provider` | — | fuerza `flux`, `pollinations` o `horde` |
| `cache=0` | desactivada | cache activa (por defecto esta apagada en el demo) |
| `format=json` | — | devuelve JSON con `image_url` |
| `b64=1` | — | devuelve JSON con la imagen en base64 |
| `hf_token` | `HF_TOKEN` | token de HuggingFace solo para ese pedido |
| `pollinations_token` | `POLLINATIONS_TOKEN` | token de Pollinations solo para ese pedido |
| `horde_token` | `HORDE_API_KEY` | apikey de Horde solo para ese pedido |

En `POST /generate` se acepta un JSON con los mismos campos.

El techo de pixeles lo pone `MAX_PIXELS` (por defecto 2300000, que es lo que acepta la API
nueva de Pollinations). Si pides algo mas grande, se achica manteniendo la proporcion.

## Endpoints

- `GET|POST /generate` — genera o sirve desde cache
- `GET /image/{id}` — sirve una imagen ya generada
- `GET /health` — estado, proveedores y cache
- `GET /stats` — contadores

## Cache

La clave es la combinacion de prompt, tamano, seed y proveedor forzado. La misma peticion
no se vuelve a generar: sale del disco en menos de 1 ms. Sobrevive a los reinicios.
Se controla con `CACHE_TTL` en segundos. Con `cache=0` se saltea la lectura y tambien la
escritura: cada pedido genera una imagen nueva y no deja nada en disco.

## API key

Si defines `API_KEY`, `/generate` la exige. Se puede pasar por el header `X-API-Key`
o por el parametro `key`. `/health` y `/stats` quedan publicos.

## Notas

- Sin `HF_TOKEN`, FLUX se agota rapido por la cuota de ZeroGPU. Sacas uno gratis en
  https://huggingface.co/settings/tokens y lo pones en el `.env`.
- Sin `POLLINATIONS_TOKEN`, ese proveedor tiene un tope de 1 imagen cada 45 segundos
  por IP. El servicio lo respeta y espera solo.
- Los proveedores son servicios de terceros: si cambian sus condiciones, el failover
  cubre el golpe pero no es garantia de disponibilidad.
