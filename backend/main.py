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
    # Realce térmico próprio. A entrada é BT física em Kelvin.
    stops = np.array([
        [185.0, 255, 255, 255], [190.0, 245, 190, 255],
        [195.0, 190, 80, 220], [200.0, 110, 35, 190],
        [205.0, 55, 35, 160], [210.0, 35, 75, 190],
        [215.0, 35, 145, 220], [220.0, 45, 200, 210],
        [225.0, 55, 180, 125], [230.0, 150, 205, 65],
        [235.0, 235, 220, 55], [240.0, 245, 155, 40],
        [245.0, 220, 75, 35], [250.0, 170, 35, 45],
        [260.0, 105, 25, 35], [270.0, 55, 25, 30],
        [280.0, 30, 25, 30], [290.0, 20, 20, 25],
        [300.0, 12, 14, 20], [315.0, 5, 8, 12],
    ], dtype=np.float32)
    values = np.clip(bt, stops[0, 0], stops[-1, 0])
    rgb = np.stack(
        [np.interp(values, stops[:, 0], stops[:, channel]) for channel in (1, 2, 3)],
        axis=-1,
    )
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
            "enhancement": "CLOUDGOD Color Enhanced IR",
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
