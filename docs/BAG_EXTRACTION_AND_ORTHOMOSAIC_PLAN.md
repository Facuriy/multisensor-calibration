# Plan de extraccion, coregistro y ortomosaicos

## Objetivo final

Extraer todos los bags de todos los experimentos de forma ordenada, con imagenes
por sensor, metadata temporal/GPS y productos listos para QGIS.

La salida ideal por experimento/plot es:

- RGB ortomosaic;
- VIS ortomosaic;
- NIR ortomosaic;
- Thermal Celsius ortomosaic con escala;
- soil cover desde RGB/GLI;
- Ouster intensity;
- Ouster depth/height visual;
- metadata CSV;
- GeoJSON con ubicaciones de frames;
- masks de validez/interseccion comun.

## Extraccion base

Script preparado:

```powershell
python src/extraction/extract_all_bag_images.py `
  --bag-root X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\20260601 `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out runs/extracted_20260601 `
  --write-raw-npy `
  --write-geojson
```

Prueba rapida sin escribir datos:

```powershell
python src/extraction/extract_all_bag_images.py `
  --bag-root X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\20260601 `
  --out runs/extracted_20260601_dryrun `
  --topics rgb vis nir thermal_c ouster_intensity ouster_range `
  --limit-bags 1 `
  --dry-run
```

El script organiza:

```text
runs/extracted_YYYYMMDD/
  metadata/
    frames.csv
    frames.geojson
  images/
    BAG_NAME/
      rgb/
      vis/
      nir/
      thermal_c/
      ouster_intensity/
      ouster_range/
  raw/
    BAG_NAME/
      ...
```

## Preprocesamiento obligatorio antes de extraccion real

Antes de procesar experimentos completos hay dos piezas que deben quedar
definidas como parte del pipeline, aunque la primera puede implementarse al
final.

### 1. Calibracion radiometrica con panel de referencia

En la campania real aparece siempre un panel de reflectancia/radiometria dentro
de la imagen. Ese panel debe usarse para normalizar la respuesta de las camaras
y hacer productos comparables entre frames, bags y condiciones de iluminacion.

Estado recomendado:

```text
status: TODO antes de producto final, no bloquea el desarrollo geometrico
```

Estrategias aceptadas:

- deteccion automatica usando la posicion esperada del panel en el rig;
- reutilizar una ROI fija si el panel aparece siempre en posicion estable;
- fallback manual con una ventana de anotacion para marcar las esquinas/ROI;
- guardar la ROI por bag/frame en metadata para poder auditarla.

Salida esperada:

```text
radiometric_panel_roi_px
radiometric_panel_stats por sensor/banda
reflectance/temperature normalization coefficients
quality flag: auto/manual/missing/invalid
```

Notas:

```text
No mezclar esta calibracion con el coregistro geometrico.
La ROI del panel debe detectarse o marcarse en las imagenes originales sin
recortar ni warpear. Luego se calculan/aplican los coeficientes radiometricos
en coordenadas originales, y recien despues se hace el coregistro, crop comun y
mosaico.
```

### 2. Area de interseccion comun entre sensores

Estado:

```text
implemented: src/registration/coregister_rgb_master.py
validated on 20260623 calibration review frames
```

Para productos finales, cada frame debe recortarse a una zona donde todas las
capas validas tengan cobertura correcta:

```text
RGB reference crop
VIS warped to RGB valid mask
NIR warped to RGB valid mask
Thermal warped to RGB valid mask
Ouster depth/intensity valid mask, cuando aplique
```

La mascara final debe ser:

```text
common_valid_mask = RGB_crop
                  & VIS_valid
                  & NIR_valid
                  & Thermal_valid
                  & optional_Ouster_valid
```

Para ortomosaicos/hypercubos no conviene usar todo el frame RGB si VIS/NIR
estan cubriendo solo una franja menor. La interseccion comun evita bordes
negros, zonas extrapoladas y comparaciones pixel-a-pixel falsas.

Salida esperada:

```text
common_roi_rgb_xyxy
common_valid_mask.png
sensor_valid_masks/
cropped_registered/
metadata con area/fraccion valida
```

Validacion 20260623:

```text
frames procesados: 37 / 37
ROI comun RGB: [649, 1176, 1794, 1743]
crop: 1145 x 567 px
pixeles invalidos despues del crop: 0 en RGB/VIS/NIR/Thermal
```

Decision de diseno:

```text
RGB sigue siendo el sistema maestro.
El crop comun se define en coordenadas RGB despues de warpear VIS/NIR/Thermal.
Para Ouster se guarda una mascara separada porque su cobertura puede ser sparse.
```

## Coregistro

Despues de tener la calibracion 20260623:

```text
VIS, NIR, Thermal -> RGB usando homografias target-plane RGB-master
Ouster -> RGB usando la extrinseca fisica candidata
```

Para producto rapido:

```text
RGB como referencia comun
VIS/NIR/Thermal registrados a RGB
Ouster depth/intensity proyectado a RGB
```

Para producto fisico riguroso:

```text
Ouster -> cada camara directamente con T_camera_lidar / T_camera_rig
reemplazar homografias 2D por extrinsecas fisicas cuando haga falta escena 3D
```

## Orthomosaic

Script RGB-master actual:

```powershell
python src/extraction/make_rgb_master_multisensor_orthomosaic.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --plot-id 17 `
  --center-ns 1780318958776849920 `
  --window-ms 7000 `
  --frames 0 `
  --out runs/orthomosaic_rgb_master_plot17_20260601_allframes `
  --splat-radius 11
```

Este script usa:

```text
intrinsecas RGB + distorsion
homografias VIS/NIR/Thermal -> RGB
extrinseca Ouster -> RGB
crop comun RGB-master
GPS/GPKG para seleccionar frames por plot
RGB feature homography / soil feature homography / ECC / phase fallback para trayectoria local
```

Salidas por plot:

```text
rgb.jpg
vis.jpg
nir.jpg
soil_cover.jpg
thermal_celsius_color.jpg
depth_mm_color.jpg
height_m_color.jpg
intensity_color.jpg
depth_mm.png
height_m_float.npy
thermal_celsius_float.npy
camera_valid_mask.png
ouster_valid_mask.png
multisensor_orthomosaic_sheet.jpg
mosaic_internal_summary.json
rgb_master_multisensor_orthomosaic_summary.json
qgis/*.tif
qgis/plot_footprint_wgs84.geojson
qgis/plot_footprint_32632.geojson
qgis/qgis_export_metadata.json
```

Interpretacion:

```text
Camera layers: densas dentro del crop comun RGB-master.
Ouster layers: metricas donde ouster_valid_mask > 0; sparse/masked por naturaleza.
depth_mm.png: profundidad en milimetros donde hay retorno valido.
trajectory_quality.recommended_use:
  production_candidate = sin fallbacks y trayectoria visual fuerte
  visual_review_only   = salida util para inspeccion, no para medicion final
```

Validaciones iniciales 20260601:

```text
plot 17 all frames: 26 frames, reliable_fraction 1.0, production_candidate
plot 18 all frames: 46 frames, reliable_fraction 1.0, production_candidate
plot 24 8 frames:   reliable_fraction 0.0, visual_review_only
```

Batch runner:

```powershell
python src/extraction/run_rgb_master_orthomosaic_batch.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out-root runs/orthomosaic_rgb_master_batch_20260601 `
  --sample-windows 10 `
  --window-ms 7000 `
  --min-frames 12 `
  --frames 0 `
  --splat-radius 11
```

Batch 20260601 / bag `2026-06-01-14-59-48.bag`:

```text
plot 17: 34 frames, production_candidate, QGIS GeoTIFFs written
plot 18: 52 frames, production_candidate, QGIS GeoTIFFs written
plot 24: 56 frames, production_candidate, QGIS GeoTIFFs written
```

Nota sobre GeoTIFF/QGIS:

```text
Los GeoTIFF actuales usan georef_method=approx_plot_bounds.
Todas las capas de un plot comparten extent CRS EPSG:32632 derivado del GPKG.
Esto es suficiente para abrir/comparar en QGIS y para productos preliminares,
pero no es todavia una ortofoto metrica absoluta SLAM/bundle-adjusted.
```

## Modo bajo consumo recomendado

Cuando la PC esta ocupada, no lanzar el batch completo con todos los frames y
LiDAR. Trabajar en tres pasos pequenos:

### 1. Dry-run de descubrimiento

Solo escanea GNSS/GPKG y sincronizacion. No escribe mosaicos pesados.

```powershell
python src\extraction\run_rgb_master_orthomosaic_batch.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out-root runs\orthomosaic_rgb_master_lowmem_20260601_dryrun `
  --sample-windows 6 `
  --window-ms 5000 `
  --min-frames 8 `
  --frames 6 `
  --max-plots 3 `
  --skip-lidar `
  --no-geotiff `
  --dry-run
```

Resultado validado el 2026-06-29:

```text
plots encontrados: 17, 18, 24
plot 17: 34 frames en ventana de 5 s
plot 18: 23 frames en ventana de 5 s
plot 24: 40 frames en ventana de 5 s
```

### 2. Preview liviano camera-only

Sirve para revisar RGB/VIS/NIR/Thermal/soil cover sin cargar Ouster.

```powershell
python src\extraction\run_rgb_master_orthomosaic_batch.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out-root runs\orthomosaic_rgb_master_lowmem_camera_preview_20260629 `
  --sample-windows 6 `
  --window-ms 5000 `
  --min-frames 8 `
  --frames 6 `
  --plots 17 `
  --max-plots 1 `
  --skip-lidar `
  --no-geotiff `
  --child-timeout-sec 300
```

Salida validada:

```text
runs/orthomosaic_rgb_master_lowmem_camera_preview_20260629/plot_17/
trajectory_quality.reliable_fraction = 1.0
recommended_use = production_candidate
```

### 3. Preview liviano con LiDAR

Incluye depth/height/intensity, pero solo con pocos frames y sin GeoTIFF.

```powershell
python src\extraction\make_rgb_master_multisensor_orthomosaic.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --plot-id 17 `
  --center-ns 1780318962714678272 `
  --window-ms 3000 `
  --frames 3 `
  --out runs\orthomosaic_rgb_master_lowmem_lidar_preview_plot17_20260629 `
  --max-sync-ms 900 `
  --max-range-m 5 `
  --splat-radius 7 `
  --margin-px 2 `
  --no-geotiff
```

Salida validada:

```text
runs/orthomosaic_rgb_master_lowmem_lidar_preview_plot17_20260629/multisensor_orthomosaic_sheet.jpg
candidate_frames = 24
used_frames = 3
depth/height/intensity Ouster visibles
```

### 4. Batch completo cuando la PC este libre

Usar solo cuando haya CPU/RAM disponible:

```powershell
python src\extraction\run_rgb_master_orthomosaic_batch.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out-root runs\orthomosaic_rgb_master_batch_20260601_full `
  --sample-windows 12 `
  --window-ms 7000 `
  --min-frames 12 `
  --frames 0 `
  --splat-radius 11
```

Regla practica:

```text
debug visual:       --frames 3..6, --no-geotiff, opcional --skip-lidar
producto preliminar: --frames 8..16, LiDAR activado, --no-geotiff si solo se revisa JPG
producto QGIS:      GeoTIFF activado, LiDAR activado, correr cuando la PC este libre
producto metrico:   reemplazar trayectoria visual local por pose(t) FAST-LIO/GPS/pose-graph
```

Preset reproducible:

```powershell
tools\run_low_resource_orthomosaic_20260601.ps1 -Mode dryrun
tools\run_low_resource_orthomosaic_20260601.ps1 -Mode camera -PlotId 17 -Frames 6
tools\run_low_resource_orthomosaic_20260601.ps1 -Mode lidar -PlotId 17
```

## Corrida QGIS liviana - 2026-06-30

Objetivo: generar productos abribles en QGIS mientras la PC esta ocupada,
separando claramente los mosaicos de camaras del producto FAST-LIO/GPS.

### Ortomosaico RGB-master bajo consumo, 3 plots

Comando ejecutado:

```powershell
python src\extraction\run_rgb_master_orthomosaic_batch.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --gpkg N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg `
  --out-root runs\orthomosaic_rgb_master_lowmem_qgis_3plots_20260630 `
  --sample-windows 6 `
  --window-ms 5000 `
  --min-frames 8 `
  --frames 6 `
  --plots 17 18 24 `
  --max-plots 3 `
  --skip-lidar `
  --child-timeout-sec 300
```

Salida principal:

```text
runs\orthomosaic_rgb_master_lowmem_qgis_3plots_20260630
```

Resultados:

```text
plot 17: production_candidate, reliable_fraction 1.0
plot 18: production_candidate, reliable_fraction 1.0
plot 24: visual_review_only, reliable_fraction 0.0, fallback_pairs 5
```

Capas para abrir en QGIS:

```text
runs\orthomosaic_rgb_master_lowmem_qgis_3plots_20260630\plot_17\qgis\
runs\orthomosaic_rgb_master_lowmem_qgis_3plots_20260630\plot_18\qgis\
runs\orthomosaic_rgb_master_lowmem_qgis_3plots_20260630\plot_24\qgis\
```

Capas utiles por plot:

```text
rgb.tif
vis.tif
nir.tif
thermal_celsius_float.tif
thermal_celsius_color.tif
soil_cover.tif
camera_valid_mask.tif
```

Nota: esta corrida uso `--skip-lidar`. Por eso las capas `depth_mm`,
`height_m` e `intensity` son placeholders/no deben interpretarse como producto
LiDAR real en esta salida. Para LiDAR real usar las salidas SLAM-like previas o
la salida FAST-LIO/GPS de abajo.

### FAST-LIO/GPS georreferenciado para QGIS, plot 17

Script:

```text
src\extraction\export_fastlio_gps_qgis.py
```

Comando ejecutado:

```powershell
python src\extraction\export_fastlio_gps_qgis.py `
  --odometry-csv runs\fastlio_plot17_ouster_full\odometry.csv `
  --fastlio-bag runs\fastlio_plot17_ouster_full\fastlio_outputs.bag `
  --original-bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --out runs\fastlio_plot17_gps_qgis_20260630 `
  --max-gps-dt-ms 700 `
  --cloud-every 4 `
  --max-points-total 700000 `
  --resolution-m 0.025
```

Salida:

```text
runs\fastlio_plot17_gps_qgis_20260630
```

Metricas:

```text
CRS: EPSG:32632
odometry poses: 135
GPS fixes: 447
associated poses: 135
GPS residual mean: 0.429 m
GPS residual median: 0.358 m
GPS residual max: 1.180 m
raster points used: 534299
```

Capas QGIS:

```text
runs\fastlio_plot17_gps_qgis_20260630\fastlio_trajectory_wgs84.geojson
runs\fastlio_plot17_gps_qgis_20260630\fastlio_pose_points_wgs84.geojson
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_height_m.tif
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_intensity.tif
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_density.tif
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_height_color.tif
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_intensity_color.tif
runs\fastlio_plot17_gps_qgis_20260630\qgis\fastlio_density_color.tif
```

Advertencia tecnica: esta salida alinea FAST-LIO local con GNSS mediante una
similaridad 2D para inspeccion QGIS. Es mucho mejor que el prototipo
velocity-only, pero todavia no es el pose-graph GPS/FAST-LIO final. El siguiente
paso serio es reemplazar esta alineacion por una optimizacion con factores GPS
para producir `pose(t)` de produccion.

## SLAM / odometria / poses metricas

Estado implementado:

```text
src/extraction/decode_ouster_imu_packets.py
src/extraction/export_slam_dataset.py
src/extraction/make_lidar_slam_plot_map.py
```

El bag contiene:

```text
/ssf/os1_cloud_node/points        PointCloud2 Ouster organizado
/ssf/os1_node/imu_packets         PacketMsg Ouster IMU crudo
/ssf/gnss/fix                     NavSatFix
/ssf/gnss/vel                     TwistStamped
/ssf/gnss/time_reference          TimeReference
```

Ouster intrinsics/extrinsics para SLAM:

```text
El bag no incluye metadata Ouster completa ni beam intrinsics por-unidad.
Para la nube PointCloud2 esto no bloquea el trabajo: /ssf/os1_cloud_node/points
ya trae XYZ calculado por el driver, por lo que los beam angles ya fueron
aplicados antes de llegar a nuestro pipeline.

Frame real de la nube inspeccionado el 2026-06-27:
  /ssf/os1_cloud_node/points.header.frame_id = /os1_lidar

Para LIO se usa como semilla la extrinseca mecanica de diseno Ouster OS1:
  lidar_to_sensor_transform =
    [-1, 0, 0, 0,
      0,-1, 0, 0,
      0, 0, 1, 38.195,
      0, 0, 0, 1]  # mm

  imu_to_sensor_transform =
    [1, 0, 0, 6.253,
     0, 1, 0,-11.775,
     0, 0, 1, 7.645,
     0, 0, 0, 1]  # mm

Como el cloud esta en /os1_lidar, el YAML LIO-SAM usa:
  T_lidar_to_imu = inv(imu_to_sensor) * lidar_to_sensor
  extrinsicTrans = [-0.006253, 0.011775, 0.03055]  # m
  extrinsicRot   = diag(-1, -1, 1)
```

Decodificacion IMU validada en plot 17:

```text
samples: 1400 en ventana de 14 s
rate: ~100 Hz
accel norm mean: ~0.983 g
gyro norm mean: ~1.23 deg/s
```

Dataset SLAM exportado:

```powershell
python src/extraction/export_slam_dataset.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --out runs/slam_dataset_plot17_20260601 `
  --center-ns 1780318960714678272 `
  --window-ms 7000 `
  --npz-every 8 `
  --write-bag
```

Salida:

```text
runs/slam_dataset_plot17_20260601/slam_input.bag
runs/slam_dataset_plot17_20260601/imu.csv
runs/slam_dataset_plot17_20260601/gps.csv
runs/slam_dataset_plot17_20260601/velocity.csv
runs/slam_dataset_plot17_20260601/clouds.csv
runs/slam_dataset_plot17_20260601/lio_sam_params_seed.yaml
runs/slam_dataset_plot17_20260601/ouster_design_extrinsics.json
```

Conteos:

```text
clouds: 140
imu: 1400
gps: 35
velocity: 35
```

Pose/map local:

```powershell
python src/extraction/make_lidar_slam_plot_map.py `
  --bag X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag `
  --center-ns 1780318960714678272 `
  --window-ms 7000 `
  --out runs/slam_lidar_plot17_20260601_velocity_only `
  --every 8 `
  --voxel 0.035 `
  --max-points 12000 `
  --resolution-m 0.01 `
  --pose-mode velocity_x_minus
```

Salidas:

```text
slam_poses_local.csv
slam_poses_georef.csv
slam_trajectory_wgs84.geojson
slam_poses_wgs84.geojson
slam_topdown_intensity.jpg
slam_topdown_global_height.jpg
slam_topdown_canopy_height.jpg
qgis/slam_*.tif
```

Resultados iniciales:

```text
plot 17 ICP libre:    scale_to_GNSS 2.239, GPS residual mean 0.338 m
plot 17 ICP rigido:   scale_to_GNSS 1.290, GPS residual mean 0.340 m
plot 17 velocity:     scale_to_GNSS 1.083, GPS residual mean 0.346 m
plot 18 velocity:     scale_to_GNSS 1.629, GPS residual mean 0.191 m
plot 24 velocity:     scale_to_GNSS 0.360, GPS residual mean 0.226 m
```

Interpretacion:

```text
Ya podemos generar poses, mapas LiDAR locales, GeoJSON y GeoTIFF SLAM-like.
No podemos declarar precision milimetrica absoluta con estos resultados:
GNSS/velocity en ventanas cortas muestra inconsistencias de escala y residual
de decimetros. Para precision absoluta hace falta ejecutar un LIO/SLAM completo
con calibracion LiDAR-IMU y validar contra control/medidas externas.
```

Mientras no haya poses metricas perfectas, los mosaicos son visuales y deben
etiquetarse como aproximados. Para QGIS, cada salida debe acompaniarse con:

- CRS;
- footprint aproximado;
- frame CSV con GPS;
- transformacion pixel-mundo si se estima;
- advertencia sobre precision.

## Pendiente antes de procesamiento final

1. Implementar/recuperar deteccion del panel radiometrico.
2. Integrar `coregister_rgb_master.py` con la extraccion masiva desde bags.
3. Compensar desplazamiento/movimiento entre frames secuenciales.
4. Confirmar topic names para todos los experimentos.
5. Confirmar si el GPKG contiene `plot_id`, tratamiento, variante y geometria.
6. Reemplazar la georreferencia `approx_plot_bounds` por poses metricas
   SLAM/odometria cuando se requiera precision absoluta.
