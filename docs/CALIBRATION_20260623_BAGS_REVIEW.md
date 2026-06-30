# Revision inicial de bags de calibracion 2026-06-23

Ruta revisada:

```text
X:/PhenoRob_UAVClimate/Projects/MSP_im_Mais/UGV/BAGS/20260623
```

## Protocolo seguido

La captura sigue el protocolo definido para el rig multisensor:

- rig fijo;
- foco/zoom/resolucion sin cambios durante la sesion;
- tablero termicamente visible;
- fondo oscuro/mate;
- tres alturas: low, mid, high;
- 12 poses por altura;
- target en centro, lados, esquinas, roll y tilt.

## Manifest generado

Se genero un manifest limpio a partir del README local:

```text
data/calibration/new_session/20260623/bag_manifest_20260623.csv
data/calibration/new_session/20260623/bag_manifest_20260623.json
data/calibration/new_session/20260623/summary_20260623.json
data/calibration/new_session/20260623/checkerboard_config.json
```

## Chessboard

Informacion confirmada por el usuario:

- tablero completo: 10 x 7 cuadrados contando el borde exterior;
- patron OpenCV usable: 9 x 6 esquinas internas;
- tamano de cuadrado interno: 0.04 m;
- la ultima fila del borde exterior es algo mas pequena, por eso no debe usarse
  como restriccion metrica independiente.

## Interpretacion del README

El README contiene 38 entradas:

- 1 bag de test excluido por defecto:
  `2026-06-23-08-06-41.bag`
- 1 bag `low_left` temprano excluido por defecto porque parece repetido:
  `2026-06-23-08-30-20.bag`
- 36 bags incluidos por defecto:
  - 12 low
  - 12 mid
  - 12 high

Typos normalizados:

- `calip_` se interpreta como `calib_`.
- `low_left_02` se interpreta como repeticion/correccion de low-left.

## Chequeo de topics

Para las 36 poses incluidas por defecto:

- RGB: presente en todos los bags.
- VIS: presente en todos los bags.
- NIR: presente en todos los bags.
- Thermal Celsius: presente en todos los bags.
- GNSS: presente en todos los bags.
- Ouster cloud: presente en casi todos los bags.

Alerta:

```text
2026-06-23-08-43-50.bag / calib_p03_low_right
```

Segun el `.info`, este bag no contiene mensajes de:

```text
/ssf/os1_cloud_node/points
```

Puede usarse para intrinsecas de camaras, pero probablemente no para extrinseca
LiDAR-tablero.

## Revision visual rapida

Se genero una hoja de contacto:

```text
runs/calibration_20260623_initial_review/keypose_contactsheet.jpg
```

Observacion inicial:

- RGB, VIS, NIR y Thermal muestran el tablero claramente en poses revisadas.
- VIS/NIR tienen buen contraste de patron.
- Thermal muestra patron visible, aunque con gradientes de temperatura del
  entorno.
- Los previews Ouster crudos no estan recortados ni interpretados; para
  calibracion debe trabajarse con la nube/region del tablero.

## Scan automatico RGB + LiDAR intensity

Se ejecuto un barrido automatico con patron `9x6` y `0.04 m`.

Salidas principales:

```text
runs/calibration_20260623_checker_lidar_scan/scan_summary.json
runs/calibration_20260623_checker_lidar_scan/scan_table.csv
runs/calibration_20260623_checker_lidar_scan/combined_lidar_panel_summary.json
runs/calibration_20260623_checker_lidar_scan/combined_lidar_panel_table.csv
runs/calibration_20260623_lidar_intensity_fallback/fallback_intensity_summary.json
runs/calibration_20260623_lidar_intensity_fallback/fallback_intensity_contactsheet.jpg
```

Resultado actual para 37 bags no-test, incluyendo el `low_left` duplicado:

- RGB checker completo detectado: 30 / 37.
- LiDAR panel detectado total: 36 / 37.
- LiDAR guiado por pose RGB: 29.
- LiDAR fallback por intensidad pura: 7.
- LiDAR faltante: `calib_p03_low_right`, porque no tiene cloud topic.

El fallback LiDAR no busca en toda la imagen Ouster. Usa la zona util observada
en las detecciones correctas:

```text
Ouster intensity ROI: x = 1320..1780, y = 0..64
```

Esto evita la zona negra/no util y la parte asociada al CPU/estructura del
sensor. El fallback es util para poses donde el tablero no esta completo o el
detector RGB no encuentra las 9x6 esquinas internas.

## Refinamiento multisensor de panel/checker

Se agrego una pasada refinada con:

- normalizacion por percentiles;
- CLAHE;
- unsharp/high-pass;
- crops guiados por RGB/homografias cuando existen;
- fallback por contraste/morfologia para panel;
- control de orden/orientacion de esquinas;
- deteccion profunda Photonfocus con patrones `9x6`, `9x5` y `8x5`.

Salidas:

```text
runs/calibration_20260623_refined_multisensor_detection/
runs/calibration_20260623_deep_photonfocus_detection/
runs/calibration_20260623_final_detection_inventory/
```

Inventario final:

```text
runs/calibration_20260623_final_detection_inventory/final_detection_inventory.csv
runs/calibration_20260623_final_detection_inventory/final_detection_inventory_summary.json
```

Resumen final:

- RGB checker mejor disponible: 32 / 37.
- RGB panel: 37 / 37.
- VIS panel: 37 / 37.
- NIR panel: 37 / 37.
- Thermal panel: 37 / 37.
- LiDAR panel: 36 / 37.
- VIS checker profundo: 8 / 37.
- NIR checker profundo: 7 / 37.

Patrones Photonfocus detectados:

```text
VIS:
  9x6: 2
  9x5: 3
  8x5: 3

NIR:
  9x5: 5
  8x5: 2
```

Interpretacion:

- El target esta localizado en todos los sensores de imagen.
- Para RGB hay suficientes detecciones para calcular intrinsecas.
- Para LiDAR hay suficientes paneles para extrinseca Ouster-RGB.
- VIS/NIR tienen panel localizado en todos los bags, pero pocas detecciones de
  checker completo regular. El patron parcial `9x5` aparece de forma real y
  consistente; probablemente la ultima fila irregular/mas pequena rompe la
  deteccion `9x6`.
- Thermal sirve en esta sesion como panel/registro, no como checker preciso.

Recomendacion tecnica:

- Usar RGB + LiDAR como calibracion fisica fuerte inmediata.
- Usar VIS/NIR con detecciones `9x5/9x6` solo como dataset parcial o para
  validar homografias, no como unica base de intrinsecas definitivas.
- Mantener Thermal como panel/ROI para extrinseca/homografia, salvo que se
  haga una captura termica mas controlada.

## Preguntas pendientes

1. Revisar la galeria completa y decidir si `2026-06-23-08-30-20.bag` debe
   descartarse o si el nombre del README esta mal.
2. Confirmar si existe algun bag repetido adicional no indicado en el README.

## Intrinseca RGB candidata

Se calculo una intrinseca RGB desde las detecciones guardadas, sin volver a
leer los bags:

```text
src/calibration/calibrate_rgb_intrinsics_from_detections.py
runs/calibration_20260623_rgb_intrinsics_from_detections/
data/calibration/new_session/20260623/rgb_intrinsics_20260623.json
```

Dataset usado:

- 32 vistas RGB unicas con checker `9x6`.
- Tamano de imagen: `2448 x 2048`.
- Patron: `9 x 6` esquinas internas.
- Cuadrado interno: `0.04 m`.

Se compararon varios modelos. El modelo libre daba un RMS algo menor, pero
estimaba un punto principal vertical no fisico (`cy ~= 642 px`). Por eso se
selecciono un modelo mas estable:

```text
selected_model: fixed_pp_zero_tangent_fix_k3
fx: 3783.531381
fy: 3781.915646
cx: 1224.000000
cy: 1024.000000
dist: [-0.04293757, 0.59964374, 0, 0, 0]
RMS: 1.7456 px
views: 32
```

Interpretacion:

- Esta intrinseca es una candidata robusta para `solvePnP` y extrinseca.
- La diferencia de RMS frente al modelo libre es pequena.
- No debe considerarse definitiva hasta validar overlays/proyecciones.

## Extrinseca Ouster-RGB multipose candidata

Se preparo un refinamiento 6DoF nuevo para esta sesion:

```text
src/calibration/refine_20260623_ouster_rgb_multipose_6dof.py
runs/calibration_20260623_ouster_rgb_multipose_6dof/
```

Convencion:

```text
X_rgb = T_cam_lidar @ X_lidar_homogeneous
```

Entrada:

- 32 detecciones RGB.
- 36 planos LiDAR.
- 31 pares RGB+LiDAR validos.
- 30 pares usados tras rechazo robusto.
- Pose rechazada: `calib_p09_low_bottomright`.

Matriz candidata:

```text
-0.9992193212  0.0052440392  0.0391567130  0.0506911581
 0.0389818448 -0.0300901895  0.9987867622  0.1229994201
 0.0064159098  0.9995334314  0.0298622764 -0.1035363600
 0.0000000000  0.0000000000  0.0000000000  1.0000000000
```

Comparacion geometrica sobre las 30 poses usadas:

```text
base previous multipose:
  corner median: 80.2 mm
  pose RMS median: 86.7 mm
  normal median: 9.12 deg

20260623 multipose 6DoF:
  corner median: 5.8 mm
  pose RMS median: 7.5 mm
  normal median: 3.70 deg
```

Validacion cruzada 5-fold:

```text
valid corner median: 5.0, 5.8, 6.7, 6.0, 9.7 mm
valid normal median: 2.08, 3.32, 3.95, 4.39, 7.25 deg
```

La misma solucion aparece al iniciar desde tres matrices historicas distintas
(diferencia maxima entre matrices optimizadas menor que `2.4e-5`), por lo que
no parece depender del punto inicial.

## Paquete activo candidato

Se escribio un paquete autocontenido:

```text
data/calibration/new_session/20260623/calibration_20260623_candidate.json
data/calibration/active_calibration.json
```

Estado:

```text
candidate_needs_visual_overlay_validation
```

Siguiente paso tecnico:

- Renderizar overlays densos LiDAR sobre RGB usando esta matriz.
- Revisar visualmente tablero, brazo/plano y escenas de campo.
- Despues componer RGB -> VIS/NIR/Thermal con las homografias existentes como
  producto 2D pragmatico.

## Validacion visual Ouster-RGB

Se agrego un renderer de validacion visual:

```text
src/calibration/render_20260623_ouster_rgb_overlay_validation.py
runs/calibration_20260623_overlay_validation/
```

Salida principal:

```text
runs/calibration_20260623_overlay_validation/overlay_validation_contactsheet.jpg
runs/calibration_20260623_overlay_validation/overlay_validation_summary.json
```

La hoja compara, para seis poses, la matriz multipose anterior contra la nueva
candidata 20260623. Para evitar ruido visual se proyectan los puntos del ROI
LiDAR filtrados por el plano detectado del tablero, no todo el Ouster.

Resultado visual:

- La nueva matriz mejora claramente la alineacion vertical y el solape del
  plano del panel en la mayoria de poses.
- El solape con el bbox interno del checker mejora en 5 de 6 poses revisadas.
- Algunos puntos siguen cayendo a la derecha del checker interno; esto es
  esperable parcialmente porque el LiDAR ve el panel fisico completo, borde y
  soporte, mientras el bbox amarillo corresponde solo a las esquinas internas.
- La validacion visual es buena como candidata, pero aun conviene revisar con
  nubes densas/escenas de campo antes de declarar final toda la cadena.

Nota operativa:

- En esta sesion `X:` era visible para PowerShell pero no para Python/rosbags.
  Para renderizar, se copio temporalmente un subconjunto de bags a
  `runs/calibration_20260623_overlay_validation/bag_cache`.
- Ese cache temporal fue borrado despues de generar los JPG para no dejar
  varios GB innecesarios.
- Si se quiere regenerar la hoja, volver a crear ese cache o pasar
  `--bag-cache` apuntando a una carpeta local con los bags seleccionados.

## Homografias 2D hacia VIS refinadas

Se valido la cadena pragmatica:

```text
RGB -> VIS
NIR -> VIS
Thermal -> VIS
```

con las imagenes exportadas de la nueva sesion:

```text
runs/calibration_20260623_full_review/per_bag/
```

Primero se renderizo la cadena usando las homografias historicas:

```text
src/registration/render_20260623_registration_chain_validation.py
runs/calibration_20260623_registration_chain_validation/
```

Resultado:

- `NIR -> VIS` se ve razonable.
- `RGB -> VIS` tenia un desfase visible.
- En la unica pareja completa refinada `RGB/VIS 9x6`, el error historico
  `RGB -> VIS` era aprox. `13.5 px` mediana.

Luego se refinaron homografias de plano usando detecciones profundas
Photonfocus:

```text
src/registration/refine_20260623_vis_homographies.py
runs/calibration_20260623_refined_vis_homographies/homographies_20260623_to_vis.json
data/calibration/new_session/20260623/homographies_20260623_to_vis.json
```

Reglas usadas:

- Solo esquinas de checker, no cajas de panel.
- Para `9x5` y `8x5`, se eligio la submalla coherente con la homografia
  historica.
- Se rechazaron pares con error de guia alto.

Metricas:

```text
RGB -> VIS:
  pares usados: 2
  puntos: 108
  mediana historica: 16.89 px
  mediana refinada: 3.14 px

NIR -> VIS:
  pares usados: 4
  puntos: 180
  mediana historica: 7.49 px
  mediana refinada: 2.42 px
```

Thermal -> VIS se mantiene desde la homografia historica del panel de aluminio,
porque la sesion 20260623 no da checker termico subpixel confiable.

## Thermal guided panel detection

Despues de revisar que el tablero termico era visible pero debil, se hizo un
pase guiado usando la geometria ya conocida. En lugar de buscar el panel en toda
la imagen termica, se proyectaron las cajas RGB/VIS hacia Thermal con las
homografias disponibles y se busco solo dentro de esa ROI expandida.

Script y salidas:

```text
src/calibration/refine_20260623_thermal_guided_panel.py
runs/calibration_20260623_thermal_guided_panel/
runs/calibration_20260623_thermal_guided_panel/thermal_guided_detection_summary.json
runs/calibration_20260623_thermal_guided_panel/thermal_guided_review_page_01.jpg
runs/calibration_20260623_thermal_guided_panel/thermal_guided_binary_page_01.jpg
```

Mejoras aplicadas:

- ROI guiada por `RGB -> VIS` y `Thermal -> VIS` historico.
- Normalizaciones `gray`, `red`, `HSV-V`, `Lab-L`, diferencia `R-G` y contraste
  local.
- Thresholds Otsu, percentiles hot/cold, adaptive threshold y bordes.
- Morfologia opening/closing con kernels rectangulares.
- Rechazo de componentes enormes que llenan demasiado la ROI.

Resultado:

```text
panel termico detectado: 37 / 37
checker termico candidato: 8 / 37
```

Candidatos con checker termico:

```text
calib_p01_low_center     9x5
calib_p01_low_left       8x5
calib_p03_low_right      9x5
calib_p04_low_top        9x5
calib_p07_low_topright   9x5
calib_p10_low_roll_plus  9x5
calib_p13_mid_center     9x5
calib_p19_mid_topright   9x5
```

Interpretacion:

- La ROI guiada es util: el panel termico cae en la zona esperada en todas las
  poses.
- La mascara binaria no siempre representa solo el checker; en varias poses toma
  lona/fondo por gradientes termicos.
- Las esquinas verdes son buenas en algunas imagenes, pero no deben tratarse
  como verdad subpixel para calibracion fisica.

Tambien se probo estimar una homografia `Thermal -> VIS` con los candidatos
termicos y las detecciones VIS disponibles:

```text
src/calibration/refine_20260623_thermal_to_vis_from_guided.py
runs/calibration_20260623_thermal_to_vis_guided_checker/
```

Metricas diagnosticas:

```text
pares usados: 2
puntos: 90
inliers: 45
mediana historica: 22.84 px
mediana nueva: 9.63 px
```

Aunque mejora la metrica interna, usa solo dos poses y los corners termicos son
debiles. Por eso queda marcado como `diagnostic_not_active` y no reemplaza la
homografia termica activa.

### Thermal scalar recovery test

Se probo la sugerencia de invertir el colormap termico antes de detectar el
checker. El colormap detectado en los previews termicos 20260623 es `inferno`,
con residual bajo (`~5.5` a `10.7` en los casos evaluados). Esto confirma que
`BGR2GRAY` directo no es la representacion ideal para detectar el damero.

Modulo y prueba:

```text
src/calibration/thermal_checker.py
src/calibration/test_20260623_thermal_scalar_checker.py
runs/calibration_20260623_thermal_scalar_checker_test_v2/
```

Resultado sobre los 8 candidatos termicos previos:

```text
detectados con escalar recuperado: 7 / 8
fallo: calib_p01_low_left
```

Paginas de revision:

```text
runs/calibration_20260623_thermal_scalar_checker_test_v2/thermal_scalar_checker_page_01.jpg
runs/calibration_20260623_thermal_scalar_checker_test_v2/thermal_scalar_checker_page_02.jpg
```

Cada fila compara:

```text
thermal RGB coloreado
gris ingenuo
escalar inferno recuperado
flatfield + CLAHE
deteccion sobre el escalar
```

Conclusion:

- La recuperacion del escalar mejora claramente la lectura visual del tablero.
- Funciona bien como rescate/diagnostico focalizado.
- No conviene activar el barrido escalar completo por defecto en todos los bags:
  `findChessboardCornersSB` es muy lento cuando falla en ROIs termicas grandes.
- El script principal acepta `--use-scalar` si se quiere probar esa ruta lenta.

Recalculando `Thermal -> VIS` con estas esquinas escalares:

```text
runs/calibration_20260623_thermal_to_vis_scalar_checker/
pares usados: 2
puntos: 90
inliers: 42
mediana historica: 21.20 px
mediana nueva: 9.95 px
```

La metrica sigue siendo diagnostica, no activa, porque solo hay dos pares
Thermal/VIS suficientemente coherentes.

### Thermal RAW mono16 test

Tambien se probo el enfoque RAW sugerido por Claude. Los bags 20260623 tienen
dos topicos termicos utiles:

```text
/ssf/thermalgrabber_ros/image_mono16        -> thermal_raw, mono16 counts
/ssf/thermalgrabber_ros/image_deg_celsius   -> thermal_c, 32FC1 Celsius
```

El extractor general ya conoce ambos:

```text
src/extraction/extract_all_bag_images.py
```

Para probarlo se copio localmente un bag chico:

```text
runs/calibration_20260623_raw_bag_cache/2026-06-23-08-43-50.bag
```

y se extrajeron 5 frames de `thermal_raw` y `thermal_c`:

```text
runs/calibration_20260623_thermal_raw_smoke_extract/
```

Se agrego el modulo RAW:

```text
src/calibration/thermal_checker_raw.py
src/calibration/test_20260623_thermal_raw_checker.py
```

Resultado sobre `calib_p03_low_right`, usando la misma ROI guiada:

```text
thermal_raw mono16:
  frames evaluados: 5
  detectados: 3
  patron completo 9x5: frame 20
  parciales 8x5: frames 10 y 40
  fallos: frames 0 y 30

thermal_c 32FC1:
  frames evaluados: 5
  detectados: 3
  mismos frames que mono16
```

Rangos observados:

```text
mono16: aprox. 7264..7433 counts
Celsius: aprox. 17.4..24.2 C
```

Reviews:

```text
runs/calibration_20260623_thermal_raw_checker_test/thermal_raw_checker_page_01.jpg
runs/calibration_20260623_thermal_c_float_checker_test/thermal_raw_checker_page_01.jpg
```

Interpretacion:

- El camino RAW funciona y evita colormap/JPEG.
- En este ejemplo no aumento el recall frente al escalar ya recuperado del
  colormap; detecta algunos frames y falla otros aunque el tablero sea visible.
- `thermal_c` crudo y `thermal_raw` crudo se comportaron casi igual.
- La limitacion dominante parece ser calidad local del patron/blur/contraste y
  seleccion temporal del frame, no solamente profundidad de bits.
- Para produccion conviene extraer `thermal_raw` o `thermal_c` como `.npy` y
  hacer el detector sobre ese escalar, dejando el JPG solo para visualizacion.

### Thermal production low-load pass

Se preparo un pipeline de produccion menos agresivo para no saturar la PC. El
paso pesado inicial copiaba los bags desde `X:` al cache local; Python no ve
confiablemente el drive mapeado, asi que el procesamiento se hace siempre desde
cache:

```text
runs/calibration_20260623_raw_bag_cache/
```

El cache local contiene los 36 bags incluidos por defecto:

```text
36 bags
58.27 GB aprox.
```

Scripts:

```text
src/calibration/run_20260623_thermal_production.py
src/calibration/merge_20260623_thermal_candidates.py
tools/run_20260623_thermal_production.ps1
```

La version actual de `run_20260623_thermal_production.py` evita el patron que
castigaba la maquina:

- no guarda todos los frames del bag en memoria;
- conserva solo los mejores `N` frames por sensor;
- calcula calidad con una metrica barata sobre el ROI;
- usa un detector rapido antes de cualquier ruta pesada;
- permite bajar carga con `--frame-step` y `--max-quality-frames`.

Pase suave ejecutado:

```powershell
python src\calibration\run_20260623_thermal_production.py `
  --bag-cache runs\calibration_20260623_raw_bag_cache `
  --out-dir runs\calibration_20260623_thermal_production_lowload `
  --max-quality-frames 3 `
  --sensors thermal_raw thermal_c `
  --frame-step 2
```

Resultado RAW/32FC1 suave:

```text
procesados: 36 / 36
detectados: 6 / 36
```

Detecciones RAW/32FC1:

```text
calib_p01_low_center
calib_p03_low_right
calib_p07_low_topright
calib_p10_low_roll_plus
calib_p13_mid_center
calib_p19_mid_topright
```

Se hizo un pase focalizado en `calib_p04_low_top` con mas candidatos; el tablero
se ve visualmente en RAW, pero OpenCV SB no lo detecta en esa ruta. Por eso se
usa fallback escalar-colormap validado para ese caso.

Merge final de candidatos termicos:

```text
runs/calibration_20260623_thermal_candidates_merged/
```

Resultado combinado:

```text
detectados: 8 / 37
RAW/32FC1: 6
fallback escalar-colormap: 2
```

Detecciones combinadas:

```text
calib_p01_low_center     RAW
calib_p01_low_left       scalar fallback
calib_p03_low_right      RAW
calib_p04_low_top        scalar fallback
calib_p07_low_topright   RAW
calib_p10_low_roll_plus  RAW
calib_p13_mid_center     RAW
calib_p19_mid_topright   RAW
```

Diagnostico Thermal -> VIS con candidatos combinados:

```text
runs/calibration_20260623_thermal_to_vis_merged_candidates/
pares usados: 2
puntos: 90
inliers: 43
mediana historica: 21.72 px
mediana nueva: 9.67 px
estado: diagnostic_not_active
```

Conclusion de produccion:

- Para la camara termica, la fuente correcta de produccion es `thermal_raw` o
  `thermal_c`, no el JPG coloreado.
- El JPG/escalar invertido queda como fallback validado para casos puntuales.
- Thermal -> VIS sigue sin activarse como calibracion final porque solo dos
  pares Thermal/VIS son geometricamente coherentes.
- El siguiente salto de recall no parece venir de mas fuerza bruta, sino de
  `cbdetect`/detector parcial o clicker manual para frames termicos criticos.

Validacion visual refinada:

```text
runs/calibration_20260623_registration_chain_validation_refinedH/
runs/calibration_20260623_registration_chain_validation_refinedH/registration_chain_contactsheet.jpg
```

Interpretacion:

- Estas homografias son buenas para visualizacion y productos 2D sobre el plano
  del tablero/interseccion comun.
- No son extrinsecas fisicas 3D.
- Para plantas/campo con relieve, usar `Ouster -> RGB` fisico y luego
  `RGB -> VIS` como producto pragmatico, documentando que el ultimo paso es 2D.

El paquete activo ahora incluye estas homografias en:

```text
data/calibration/active_calibration.json
data/calibration/new_session/20260623/calibration_20260623_candidate.json
```

## RGB como referencia maestra

Se replanteo el registro pragmatico 2D para usar RGB como frame comun:

```text
Ouster 3D -> RGB
VIS 2D -> RGB
NIR 2D -> RGB
Thermal 2D -> RGB
```

Esto es mas coherente porque `Ouster -> RGB` ya es la extrinseca fisica 3D
mejor calibrada y RGB es la camara mas nitida.

Scripts y salidas:

```text
src/registration/build_20260623_rgb_master_homographies.py
runs/calibration_20260623_rgb_master_homographies/homographies_20260623_to_rgb.json
data/calibration/new_session/20260623/homographies_20260623_to_rgb.json

src/registration/render_20260623_rgb_master_validation.py
runs/calibration_20260623_rgb_master_validation/
```

Seleccion de matrices:

```text
VIS -> RGB:
  seleccion: inversa compuesta de RGB -> VIS
  mediana validada: 7.11 px

NIR -> RGB:
  seleccion: ajuste directo con corners
  mediana validada: 8.39 px

Thermal -> RGB:
  seleccion: Thermal -> VIS -> RGB
  estado: candidato / diagnostico
```

Validacion visual:

```text
runs/calibration_20260623_rgb_master_validation/rgb_master_contactsheet_01.jpg
runs/calibration_20260623_rgb_master_validation/rgb_master_contactsheet_02.jpg
```

El paquete activo ahora incluye:

```text
homographies_2d_to_rgb
```

Advertencia: `VIS/NIR/Thermal -> RGB` sigue siendo registro 2D planar. Para
registrar todos los pixeles de una escena 3D real con relieve se necesitan
extrinsecas fisicas por camara y/o profundidad por pixel.
