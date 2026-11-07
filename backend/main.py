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
PRODUCT_PREFIX = "ABI-L1b-RadF"
SOUTH_AMERICA = (-82.0, -56.5, -32.0, 13.5)  # west, south, east, north
WIDTH = max(300, int(os.getenv("BT_WIDTH", "1500")))
HEIGHT = max(300, int(os.getenv("BT_HEIGHT", "1540")))
CACHE_SECONDS = max(60, int(os.getenv("BT_CACHE_SECONDS", "240")))

app = FastAPI(title="CLOUDGOD GOES-19 IR Color Enhanced", version="2.0.0")
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
        "Nenhum arquivo ABI-L1b-RadF-M6C13 recente foi encontrado no bucket público NOAA "
        "nas últimas 8 horas. Verifique a publicação do produto."
    )


# Tabela oficial de cores NOAA/NESDIS/STAR para canais infravermelhos 11–15.
# A barra publicada pela NOAA contém a escala em °C (-110 a +57 °C).
NOAA_IR_COLORBAR_URL = (
    "https://www.goes.noaa.gov/images/colorbars/ColorBar450Bands11-15_horz.png"
)
_IR_LUT = None
_IR_LUT_LOCK = threading.Lock()


def load_noaa_ir_lut():
    """Baixa a barra oficial NOAA e transforma seus pixels em uma LUT contínua."""
    global _IR_LUT
    if _IR_LUT is not None:
        return _IR_LUT

    with _IR_LUT_LOCK:
        if _IR_LUT is not None:
            return _IR_LUT

        import urllib.request
        request = urllib.request.Request(
            NOAA_IR_COLORBAR_URL,
            headers={"User-Agent": "CLOUDGOD-GOES19-IR/2.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                palette_bytes = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível obter a tabela oficial IR da NOAA: {exc}"
            ) from exc

        with Image.open(io.BytesIO(palette_bytes)) as palette_image:
            bar = np.asarray(palette_image.convert("RGB"), dtype=np.float32)

        if bar.ndim != 3 or bar.shape[1] < 100:
            raise RuntimeError("A barra oficial de cores NOAA retornou formato inválido.")

        # Usa a mediana vertical para reduzir a influência de artefatos de borda.
        y0 = max(0, int(bar.shape[0] * 0.25))
        y1 = min(bar.shape[0], int(bar.shape[0] * 0.75) + 1)
        strip = np.median(bar[y0:y1, :, :], axis=0)

        # Escala impressa na própria legenda NOAA: -110 °C até +57 °C.
        # A LUT é amostrada da barra publicada, não criada por cores arbitrárias.
        source_x = np.linspace(-110.0, 57.0, strip.shape[0], dtype=np.float32)
        target_c = np.linspace(-110.0, 57.0, 2048, dtype=np.float32)
        lut = np.stack([
            np.interp(target_c, source_x, strip[:, channel])
            for channel in range(3)
        ], axis=-1)

        _IR_LUT = (target_c, np.clip(lut, 0, 255))
        log.info("Tabela IR oficial NOAA/NESDIS/STAR carregada: %s amostras", len(target_c))
        return _IR_LUT


def color_enhanced(brightness_temperature_k):
    """
    Aplica a tabela IR Color Enhanced publicada pela NOAA/NESDIS/STAR.
    A fonte física é a radiância L1b calibrada do GOES-19; a radiância é
    convertida para temperatura de brilho apenas para indexar a tabela IR.
    """
    bt_k = np.asarray(brightness_temperature_k, dtype=np.float32)
    temp_c, colors = load_noaa_ir_lut()
    bt_c = bt_k - 273.15
    valid = np.isfinite(bt_c)
    safe = np.where(valid, bt_c, 57.0)
    rgb = np.stack([
        np.interp(safe, temp_c, colors[:, channel])
        for channel in range(3)
    ], axis=-1)
    rgb[~valid] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_png():
    """Download GOES-19 L1b Band 13 radiance, calibrate, crop and warp to lat/lon."""
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
            if "Rad" not in ds.variables:
                raise RuntimeError("O NetCDF ABI-L1b não contém a variável Rad.")

            variable = ds.variables["Rad"]
            rad_units = str(getattr(variable, "units", "")).strip()
            planck_fk1 = float(getattr(ds, "planck_fk1", np.nan))
            planck_fk2 = float(getattr(ds, "planck_fk2", np.nan))
            planck_bc1 = float(getattr(ds, "planck_bc1", np.nan))
            planck_bc2 = float(getattr(ds, "planck_bc2", np.nan))
            if not all(np.isfinite(v) for v in (planck_fk1, planck_fk2, planck_bc1, planck_bc2)):
                raise RuntimeError("Coeficientes Planck ausentes no arquivo ABI-L1b.")

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
            # Calcula a janela da América do Sul e lê apenas a parte necessária de Rad.
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
                raise RuntimeError("Não foi possível recortar a janela Rad da América do Sul.")
            x0, x1 = int(ix[0]), int(ix[-1]) + 1
            y0, y1 = int(iy.min()), int(iy.max()) + 1
            rad_ma = variable[y0:y1, x0:x1]
            radiance = np.asarray(np.ma.filled(rad_ma, np.nan), dtype=np.float32)

            # ABI L1b emissive bands store spectral radiance. NetCDF4 applies
            # scale_factor/add_offset automatically when reading Rad.
            radiance[(radiance <= 0) | (~np.isfinite(radiance))] = np.nan
            denominator = np.log1p(planck_fk1 / radiance)
            bt = (planck_fk2 / denominator - planck_bc1) / planck_bc2
            bt = np.asarray(bt, dtype=np.float32)
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
        "service": "CLOUDGOD GOES-19 IR Color Enhanced",
        "status": "online",
        "source": "NOAA ABI-L1b-RadF-M6C13",
        "source_units": "spectral radiance (ABI L1b Rad)",
        "display_calibration": "Planck brightness temperature for NOAA IR enhancement table",
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
            "product": "ABI-L1b-RadF-M6C13", "variable": "Rad", "source_units": "spectral radiance",
        "display_calibration": "Planck brightness temperature for NOAA IR enhancement table",
            "scan_time": item["scan"], "source_file": item["key"],
            "enhancement": "NOAA/NESDIS/STAR official IR color-enhanced bar (bands 11-15)",
            "enhancement_table": NOAA_IR_COLORBAR_URL,
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
                "X-CLOUDGOD-Units": "ABI L1b spectral radiance",
                "X-CLOUDGOD-Enhancement": "NOAA-NESDIS-STAR-IR-COLOR-ENHANCED",
                "X-CLOUDGOD-Source": "NOAA ABI-L1b-RadF-M6C13",
                "X-CLOUDGOD-Scan-Time": str(item["scan"]),
            },
        )
    except Exception as exc:
        log.exception("Falha ao gerar raster BT")
        raise HTTPException(status_code=503, detail=f"Falha no processamento NOAA: {exc}")
