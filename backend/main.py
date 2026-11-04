"""CLOUDGOD GOES-19 ABI Band 13 brightness-temperature image service."""
import io
import os
import threading
import time
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

BUCKET = "noaa-goes19"
PRODUCT_PREFIX = "ABI-L2-CMIPF"
# WGS84 extent covering the South American continent and nearby oceans.
SOUTH_AMERICA = (-82.0, -56.5, -32.0, 13.5)  # west, south, east, north
WIDTH = max(300, int(os.getenv("BT_WIDTH", "1500")))
HEIGHT = max(300, int(os.getenv("BT_HEIGHT", "1540")))
CACHE_SECONDS = max(60, int(os.getenv("BT_CACHE_SECONDS", "240")))

app = FastAPI(title="CLOUDGOD GOES-19 Brightness Temperature", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1"))
_cache = {"at": 0.0, "png": None, "scan": None}
_lock = threading.Lock()


def latest_c13_key():
    now = datetime.now(timezone.utc)
    # Check recent UTC hours, allowing for scan and publication latency.
    for back in range(0, 5):
        stamp = now.timestamp() - back * 3600
        dt = datetime.fromtimestamp(stamp, timezone.utc)
        prefix = f"{PRODUCT_PREFIX}/{dt.year}/{dt.timetuple().tm_yday:03d}/{dt.hour:02d}/"
        pager = s3.get_paginator("list_objects_v2")
        matches = []
        for page in pager.paginate(Bucket=BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if "M6C13_G19_" in key and key.endswith(".nc"):
                    matches.append(item)
        if matches:
            return max(matches, key=lambda item: item["LastModified"])["Key"]
    raise RuntimeError("Não foi encontrado arquivo ABI-L2-CMIPF-M6C13 recente no bucket público da NOAA.")


def color_enhanced(bt):
    # Color enhancement térmico para IR de onda longa: tons quentes em
    # temperaturas altas e roxo/vermelho/branco nos topos muito frios.
    # Os valores de entrada continuam sendo BT física em Kelvin.
    stops = np.array([
        [185.0, 255, 255, 255],
        [190.0, 245, 190, 255],
        [195.0, 190, 80, 220],
        [200.0, 110, 35, 190],
        [205.0, 55, 35, 160],
        [210.0, 35, 75, 190],
        [215.0, 35, 145, 220],
        [220.0, 45, 200, 210],
        [225.0, 55, 180, 125],
        [230.0, 150, 205, 65],
        [235.0, 235, 220, 55],
        [240.0, 245, 155, 40],
        [245.0, 220, 75, 35],
        [250.0, 170, 35, 45],
        [260.0, 105, 25, 35],
        [270.0, 55, 25, 30],
        [280.0, 30, 25, 30],
        [290.0, 20, 20, 25],
        [300.0, 12, 14, 20],
        [315.0, 5, 8, 12],
    ], dtype=np.float32)
    values = np.clip(bt, stops[0, 0], stops[-1, 0])
    rgb = np.stack([np.interp(values, stops[:, 0], stops[:, channel]) for channel in (1, 2, 3)], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_png():
    key = latest_c13_key()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    raw = obj["Body"].read()
    with Dataset("inmemory.nc", mode="r", memory=raw) as ds:
        if "CMI" not in ds.variables:
            raise RuntimeError("O NetCDF da NOAA não contém a variável CMI esperada.")
        cmi = ds.variables["CMI"][:]
        bt = np.asarray(np.ma.filled(cmi, np.nan), dtype=np.float32)
        units = getattr(ds.variables["CMI"], "units", "K")
        if str(units).lower() not in ("k", "kelvin"):
            raise RuntimeError(f"Unidade inesperada para CMI: {units}. A imagem foi bloqueada para evitar BT incorreta.")
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        projection = ds.variables["goes_imager_projection"]
        h = float(projection.perspective_point_height)
        lon0 = float(projection.longitude_of_projection_origin)
        sweep = str(getattr(projection, "sweep_angle_axis", "x"))
        a = float(getattr(projection, "semi_major_axis", 6378137.0))
        b = float(getattr(projection, "semi_minor_axis", 6356752.31414))
        scan = getattr(ds, "time_coverage_end", "horário não informado")

    # Validar faixa física plausível e descartar fill values / pixels fora do disco.
    bt[(bt < 150.0) | (bt > 350.0)] = np.nan
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    src_transform = Affine(dx * h, 0, (x[0] - dx / 2) * h,
                           0, dy * h, (y[0] - dy / 2) * h)
    src_crs = CRS.from_proj4(
        f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +a={a} +b={b} +units=m +no_defs"
    )
    dst_crs = CRS.from_epsg(4326)
    west, south, east, north = SOUTH_AMERICA
    dst_transform = from_bounds(west, south, east, north, WIDTH, HEIGHT)
    warped = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    reproject(
        source=bt,
        destination=warped,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=2,
    )
    valid = np.isfinite(warped)
    if valid.sum() < 1000:
        raise RuntimeError("A reprojeção do GOES-19 não produziu pixels válidos na América do Sul.")
    rgb = color_enhanced(np.nan_to_num(warped, nan=315.0))
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(valid, 238, 0).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG", optimize=True)
    return out.getvalue(), key, scan


@app.get("/")
def health():
    return {"service": "CLOUDGOD GOES-19 BT", "status": "online", "source": "NOAA ABI-L2-CMIPF-M6C13", "units": "K"}


@app.get("/bt.json")
def metadata():
    if not _cache["png"] or time.time() - _cache["at"] > CACHE_SECONDS:
        with _lock:
            if not _cache["png"] or time.time() - _cache["at"] > CACHE_SECONDS:
                png, key, scan = make_png()
                _cache.update(at=time.time(), png=png, scan=scan, key=key)
    return {"satellite": "GOES-19", "band": 13, "wavelength_um": 10.3,
            "product": "ABI-L2-CMIPF-M6C13", "variable": "CMI", "units": "K",
            "scan_time": _cache["scan"], "source_file": _cache["key"],
            "enhancement": "CLOUDGOD Color Enhanced IR", "bounds": SOUTH_AMERICA}


@app.get("/bt.png")
def brightness_temperature():
    try:
        if not _cache["png"] or time.time() - _cache["at"] > CACHE_SECONDS:
            with _lock:
                if not _cache["png"] or time.time() - _cache["at"] > CACHE_SECONDS:
                    png, key, scan = make_png()
                    _cache.update(at=time.time(), png=png, scan=scan, key=key)
        return Response(content=_cache["png"], media_type="image/png", headers={
            "Cache-Control": "public, max-age=180",
            "X-CLOUDGOD-Satellite": "GOES-19",
            "X-CLOUDGOD-Band": "13",
            "X-CLOUDGOD-Units": "K",
            "X-CLOUDGOD-Source": "NOAA ABI-L2-CMIPF-M6C13",
        })
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
