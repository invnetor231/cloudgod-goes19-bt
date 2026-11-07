"""CLOUDGOD GOES-19 ABI Band 13 brightness-temperature image service."""
import io
import os
import logging
import threading
import time
import tempfile
from datetime import datetime, timezone

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from netCDF4 import Dataset
from PIL import Image
from rasterio import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from pyproj import Transformer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("cloudgod.bt")

BUCKET = "noaa-goes19"
PRODUCT_PREFIX = "ABI-L2-CMIPF"
SOUTH_AMERICA = (-82.0, -56.5, -32.0, 13.5)  # west, south, east, north
WIDTH = max(300, int(os.getenv("BT_WIDTH", "1500")))
HEIGHT = max(300, int(os.getenv("BT_HEIGHT", "1540")))
CACHE_SECONDS = max(60, int(os.getenv("BT_CACHE_SECONDS", "240")))

app = FastAPI(title="CLOUDGOD GOES-19 Brightness Temperature", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

s3 = boto3.client(
    "s3",
    config=Config(
        signature_version=UNSIGNED,
        region_name="us-east-1",
        connect_timeout=15,
        read_timeout=90,
        retries={"max_attempts": 3, "mode": "standard"},
        max_pool_connections=4,
    ),
)
_cache = {"at": 0.0, "png": None, "scan": None, "key": None}
_lock = threading.Lock()


def latest_c13_key():
    now = datetime.now(timezone.utc)
    pager = s3.get_paginator("list_objects_v2")
    # GOES Full Disk C13 é publicado em ciclos de 10 minutos.
    # Percorre as últimas horas para tolerar atrasos de publicação.
    for back in range(0, 9):
        dt = datetime.fromtimestamp(now.timestamp() - back * 3600, timezone.utc)
        prefix = f"{PRODUCT_PREFIX}/{dt.year}/{dt.timetuple().tm_yday:03d}/{dt.hour:02d}/"
        matches = []
        for page in pager.paginate(Bucket=BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if "M6C13_G19_" in key and key.endswith(".nc"):
                    matches.append(item)
        if matches:
            chosen = max(matches, key=lambda item: item["LastModified"])
            log.info("Arquivo NOAA selecionado: %s", chosen["Key"])
            return chosen["Key"]
    raise RuntimeError(
        "Nenhum arquivo ABI-L2-CMIPF-M6C13 recente foi encontrado no bucket público NOAA "
        "nas últimas 8 horas. Verifique a publicação do produto."
    )


def color_enhanced(bt):
    """
    IR Color Enhanced inspirado na referência enviada:
    fundo infravermelho em escala de cinza, nuvens frias em ciano/azul,
    e núcleos convectivos em verde, amarelo, laranja, vermelho e tons escuros.
    """
    bt = np.asarray(bt, dtype=np.float32)

    # Base em escala de cinza: superfícies quentes escuras e nuvens frias claras.
    gray_stops = np.array([
        [315.0, 8], [300.0, 25], [285.0, 55], [270.0, 92],
        [255.0, 135], [245.0, 175], [235.0, 205], [225.0, 230],
    ], dtype=np.float32)
    gray = np.interp(
        np.clip(bt, gray_stops[-1, 0], gray_stops[0, 0]),
        gray_stops[::-1, 0], gray_stops[::-1, 1]
    )
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    # Color enhancement para os topos de nuvens mais frios.
    # A sequência reproduz o padrão visual da referência: ciano, azul,
    # verde, amarelo, laranja, vermelho e núcleos escuros.
    stops = np.array([
        [240.0, 205, 220, 225],
        [235.0, 115, 225, 255],
        [230.0,   0, 155, 255],
        [225.0,   0,  65, 255],
        [220.0,   0,  35, 190],
        [215.0,   0, 165,  55],
        [210.0, 155, 220,   0],
        [205.0, 255, 235,   0],
        [200.0, 255, 150,   0],
        [195.0, 245,  55,   0],
        [190.0, 220,   0,   0],
        [185.0,  95,  15,  15],
        [180.0,  35,  35,  38],
        [175.0, 115, 115, 120],
        [165.0, 225, 225, 230],
    ], dtype=np.float32)

    temps = stops[:, 0][::-1]
    colors = stops[:, 1:4][::-1]
    clipped = np.clip(bt, temps[0], temps[-1])
    enhanced = np.stack([
        np.interp(clipped, temps, colors[:, channel])
        for channel in range(3)
    ], axis=-1)

    # Mistura progressiva: mantém o fundo cinza e colore apenas os
    # topos suficientemente frios, sem criar uma máscara abrupta.
    blend = np.clip((240.0 - bt) / 5.0, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - blend) + enhanced * blend
    return np.clip(rgb, 0, 255).astype(np.uint8)

def make_png():
    """Download the newest full-disk C13 file, crop before reading CMI, and warp to lat/lon."""
    key = latest_c13_key()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    raw = obj["Body"].read()
    log.info("NetCDF NOAA baixado: %.1f MB (%s)", len(raw) / 1024 / 1024, key)

    # Abrir por arquivo temporário é mais compatível com builds netCDF-C do Render
    # do que Dataset(..., memory=bytes), que pode falhar dependendo da compilação.
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(raw)
        tmp.flush()
        with Dataset(tmp.name, mode="r") as ds:
            if "CMI" not in ds.variables:
                raise RuntimeError("O NetCDF não contém a variável CMI.")
            variable = ds.variables["CMI"]
            units = str(getattr(variable, "units", "K")).strip().lower()
            if units not in ("k", "kelvin"):
                raise RuntimeError(f"Unidade CMI inesperada: {units!r}; esperado Kelvin.")

            x = np.asarray(ds.variables["x"][:], dtype=np.float64)
            y = np.asarray(ds.variables["y"][:], dtype=np.float64)
            projection = ds.variables["goes_imager_projection"]
            h = float(projection.perspective_point_height)
            lon0 = float(projection.longitude_of_projection_origin)
            sweep = str(getattr(projection, "sweep_angle_axis", "x"))
            a = float(getattr(projection, "semi_major_axis", 6378137.0))
            b = float(getattr(projection, "semi_minor_axis", 6356752.31414))
            scan = str(getattr(ds, "time_coverage_end", "horário não informado"))

            if x.size < 2 or y.size < 2:
                raise RuntimeError("Coordenadas de varredura x/y inválidas no NetCDF.")

            src_crs = CRS.from_proj4(
                f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} "
                f"+a={a} +b={b} +units=m +no_defs"
            )
            # Calcula a janela da América do Sul na projeção nativa e lê apenas
            # essa parte do CMI. Isso reduz muito o pico de RAM no Render.
            west, south, east, north = SOUTH_AMERICA
            transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
            lon_samples = np.linspace(west, east, 9)
            lat_samples = np.linspace(south, north, 9)
            xx, yy = np.meshgrid(lon_samples, lat_samples)
            px, py = transformer.transform(xx.ravel(), yy.ravel())
            finite = np.isfinite(px) & np.isfinite(py)
            if not finite.any():
                raise RuntimeError("A área solicitada está fora da visada do GOES-19.")
            # x/y do ABI são ângulos em radianos; a projeção usa metros.
            x_m = x * h
            y_m = y * h
            xmin, xmax = float(np.min(np.asarray(px)[finite])), float(np.max(np.asarray(px)[finite]))
            ymin, ymax = float(np.min(np.asarray(py)[finite])), float(np.max(np.asarray(py)[finite]))
            ix = np.where((x_m >= xmin - 180000) & (x_m <= xmax + 180000))[0]
            iy = np.where((y_m >= ymin - 180000) & (y_m <= ymax + 180000))[0]
            if ix.size < 2 or iy.size < 2:
                raise RuntimeError("Não foi possível recortar a janela CMI da América do Sul.")
            x0, x1 = int(ix[0]), int(ix[-1]) + 1
            y0, y1 = int(iy.min()), int(iy.max()) + 1
            cmi = variable[y0:y1, x0:x1]
            bt = np.asarray(np.ma.filled(cmi, np.nan), dtype=np.float32)
            x_crop = x[x0:x1]
            y_crop = y[y0:y1]

    bt[(bt < 150.0) | (bt > 350.0)] = np.nan
    dx, dy = float(np.median(np.diff(x_crop))), float(np.median(np.diff(y_crop)))
    src_transform = Affine(
        dx * h, 0, (x_crop[0] - dx / 2) * h,
        0, dy * h, (y_crop[0] - dy / 2) * h,
    )
    dst_transform = from_bounds(west, south, east, north, WIDTH, HEIGHT)
    warped = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    reproject(
        source=bt,
        destination=warped,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=CRS.from_epsg(4326),
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=2,
    )
    valid = np.isfinite(warped)
    valid_count = int(valid.sum())
    if valid_count < 1000:
        raise RuntimeError(
            f"Reprojeção sem cobertura suficiente na América do Sul ({valid_count} pixels válidos)."
        )
    rgb = color_enhanced(np.nan_to_num(warped, nan=315.0))
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(valid, 238, 0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(output, format="PNG", optimize=True)
    log.info(
        "PNG gerado: %sx%s, %s pixels válidos, janela ABI %sx%s, scan=%s",
        WIDTH, HEIGHT, valid_count, x1 - x0, y1 - y0, scan,
    )
    return output.getvalue(), key, scan


def get_cached_product():
    if _cache["png"] is None or time.time() - _cache["at"] > CACHE_SECONDS:
        with _lock:
            if _cache["png"] is None or time.time() - _cache["at"] > CACHE_SECONDS:
                png, key, scan = make_png()
                _cache.update(at=time.time(), png=png, scan=scan, key=key)
    return _cache


@app.get("/")
@app.get("/health")
def health():
    return {
        "service": "CLOUDGOD GOES-19 BT",
        "status": "online",
        "source": "NOAA ABI-L2-CMIPF-M6C13",
        "units": "K",
        "image_endpoint": "/bt.png",
        "metadata_endpoint": "/bt.json",
        "version": app.version,
    }


@app.get("/bt.json")
def metadata():
    try:
        item = get_cached_product()
        return {
            "satellite": "GOES-19", "band": 13, "wavelength_um": 10.3,
            "product": "ABI-L2-CMIPF-M6C13", "variable": "CMI", "units": "K",
            "scan_time": item["scan"], "source_file": item["key"],
            "enhancement": "GOES-19 Band 13 IR Color Enhanced",
            "bounds": SOUTH_AMERICA,
            "image_size": [WIDTH, HEIGHT],
            "cache_age_seconds": round(time.time() - item["at"], 1),
        }
    except Exception as exc:
        log.exception("Falha ao gerar metadados BT")
        raise HTTPException(status_code=503, detail=f"Falha no processamento NOAA: {exc}")


@app.get("/bt.png")
def brightness_temperature():
    try:
        item = get_cached_product()
        return Response(
            content=item["png"],
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=180",
                "Access-Control-Allow-Origin": "*",
                "X-CLOUDGOD-Satellite": "GOES-19",
                "X-CLOUDGOD-Band": "13",
                "X-CLOUDGOD-Units": "K",
                "X-CLOUDGOD-Source": "NOAA ABI-L2-CMIPF-M6C13",
                "X-CLOUDGOD-Scan-Time": str(item["scan"]),
            },
        )
    except Exception as exc:
        log.exception("Falha ao gerar raster BT")
        raise HTTPException(status_code=503, detail=f"Falha no processamento NOAA: {exc}")
