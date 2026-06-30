param(
    [ValidateSet("dryrun", "camera", "lidar")]
    [string]$Mode = "dryrun",
    [string]$PlotId = "17",
    [int]$Frames = 6,
    [string]$OutRoot = "runs\orthomosaic_rgb_master_lowmem_manual_20260601"
)

$ErrorActionPreference = "Stop"

$bag = "X:\PhenoRob_UAVClimate\Projects\MSP_im_Mais\UGV\BAGS\20260601\2026-06-01-14-59-48.bag"
$gpkg = "N:\Sensorik\02_Projekte\2026\SE08\versuchslayout.gpkg"
$centerNsPlot17 = "1780318962714678272"

if ($Mode -eq "dryrun") {
    python src\extraction\run_rgb_master_orthomosaic_batch.py `
        --bag $bag `
        --gpkg $gpkg `
        --out-root $OutRoot `
        --sample-windows 6 `
        --window-ms 5000 `
        --min-frames 8 `
        --frames $Frames `
        --max-plots 3 `
        --skip-lidar `
        --no-geotiff `
        --dry-run
    exit $LASTEXITCODE
}

if ($Mode -eq "camera") {
    python src\extraction\run_rgb_master_orthomosaic_batch.py `
        --bag $bag `
        --gpkg $gpkg `
        --out-root $OutRoot `
        --sample-windows 6 `
        --window-ms 5000 `
        --min-frames 8 `
        --frames $Frames `
        --plots $PlotId `
        --max-plots 1 `
        --skip-lidar `
        --no-geotiff `
        --child-timeout-sec 300
    exit $LASTEXITCODE
}

if ($Mode -eq "lidar") {
    if ($PlotId -ne "17") {
        throw "The lightweight direct LiDAR preset currently has a validated center timestamp only for plot 17."
    }
    python src\extraction\make_rgb_master_multisensor_orthomosaic.py `
        --bag $bag `
        --gpkg $gpkg `
        --plot-id $PlotId `
        --center-ns $centerNsPlot17 `
        --window-ms 3000 `
        --frames 3 `
        --out $OutRoot `
        --max-sync-ms 900 `
        --max-range-m 5 `
        --splat-radius 7 `
        --margin-px 2 `
        --no-geotiff
    exit $LASTEXITCODE
}
