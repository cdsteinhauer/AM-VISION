# Robot Vision

Browser-based part inspection app for an Astra Plus Pro depth camera on a Jetson Orin Nano. The project is packaged as a ROS2 Python package so it can live under:

`/home/csteinhauer/robot_vision`

The first MVP includes a FastAPI service, SICK-style browser UI, mock/OpenCV/camera-provider abstraction, freeform inspection tools, calibration, manual and auto snapshot triggering, and saved inspection reports.

## Local PC Development

From this folder:

```powershell
python -m pip install -e ".[test]"
python -m pytest
robot-vision-inspect-sample
python -m robot_vision --mock --host 127.0.0.1 --port 8080
```

Open:

`http://127.0.0.1:8080`

## Jetson One-Click Startup

The Jetson deployment installs:

- `/home/csteinhauer/robot_vision/start_robot_vision.sh`
- `/home/csteinhauer/Desktop/Start Robot Vision.desktop`

Double-click `Start Robot Vision` on the Jetson desktop. It restarts the hardware camera web server, verifies `/api/health`, and opens:

`http://127.0.0.1:8080`

For the full RGB+depth hardware path, the launcher also starts:

```bash
source /opt/ros/humble/setup.bash
source /home/csteinhauer/astra_ws/install/setup.bash
ros2 launch astra_camera astra.launch.py
```

Robot Vision then subscribes to the Astra ROS image topics.

## Jetson Deploy

Default SSH target:

- `csteinhauer@jetson.local`
- fallback: `csteinhauer@192.168.20.241`
- port: `22`
- remote app folder: `/home/csteinhauer/robot_vision`
- ROS domain: `77`

Dry run:

```powershell
.\scripts\deploy_to_jetson.ps1 -DryRun
```

Copy and build:

```powershell
.\scripts\deploy_to_jetson.ps1
```

If this PC does not have SSH key auth set up and you need to type the Jetson password:

```powershell
.\scripts\deploy_to_jetson.ps1 -AllowPasswordPrompt
```

Preferred one-time setup for non-interactive deploys:

```powershell
.\scripts\setup_jetson_ssh_key.ps1
.\scripts\deploy_to_jetson.ps1
```

On the Jetson:

```bash
cd /home/csteinhauer/robot_vision
export ROS_DOMAIN_ID=77
python3 -m pip install --user -r requirements.txt
python3 -m robot_vision --config config/app.yaml --mock --host 0.0.0.0 --port 8080
```

Then open:

`http://jetson.local:8080`

## Camera Modes

Set `config/app.yaml`:

```yaml
camera:
  provider: mock
```

Supported providers:

- `mock`: generated rectangle/depth scene for development and tests.
- `opencv`: standard USB camera via OpenCV/V4L2.
- `ros_astra`: RGB and depth from the existing ROS2 Astra driver topics.
- `astra_hybrid`: RGB from `/dev/video0` via OpenCV/V4L2 and depth from `/camera/depth/image_raw`. This is the current Jetson hardware path.
- `orbbec`, `astra`, `astra_plus_pro`: SDK-backed Orbbec provider. The Jetson default is now the Orbbec Femto RGB + Depth mode.

The SDK-backed Orbbec provider is intentionally isolated. If `pyorbbecsdk` is missing or incompatible on the Jetson, the app still runs in mock/OpenCV/Astra fallback modes while the SDK install is resolved. Install the Python binding with `python3 -m pip install --user pyorbbecsdk2`.

Current Jetson hardware status:

- Astra RGB is active through `/dev/video0`.
- Orbbec Femto is the default app camera path through the Orbbec SDK.
- Astra depth remains available through the existing `/home/csteinhauer/astra_ws` ROS2 Astra driver on `/camera/depth/image_raw` when Astra mode is selected/configured.

## Inspection Workflow

1. Open the browser UI.
2. Select the viewer mode:
   - `Live View`: lightweight RGB/depth video preview with no inspection or trigger processing.
   - `Live Capture`: preview plus part-presence trigger processing for auto capture.
   - `Snap Capture`: no automatic refresh; each `Snap` press grabs and displays a new still frame.
3. Create or edit inspection tools in the recipe panel.
4. Save calibration using a known plate/grid/rectangle.
5. With the fixture empty, use `Capture Depth Zero` to save the table/fixture plane used for depth height.
   After depth zero is saved, rectangle and edge tools prefer depth-difference part detection before falling back to RGB contrast.
   If the green depth-detected overlay is offset from the RGB part, adjust `Depth X px` and `Depth Y px`, then save depth alignment.
6. Press `Inspect`, or use `Live Capture` with `Auto`.
   Use `Cycle ms` to set the Live Capture preview/trigger interval. The result banner keeps a rolling average and standard deviation for incoming W/H/Z measurements.
7. Review pass/fail, dimensions, overlays, and saved reports.

Reports are written under:

`data/reports/<timestamp-id>/`

Each report stores:

- `rgb.png`
- `depth.png` when available
- `overlay.png`
- `result.json`

## Hugging Face Vision Training

Saved reports can be used to train a PASS/FAIL image classifier. Each report contributes:

- `rgb.png` as the training image
- `result.json -> result.passed` as the label

Install optional training dependencies on the machine that will train the model:

```bash
python -m pip install -e ".[train]"
```

On Jetson, `torch` and `torchvision` must be compatible with the installed JetPack/L4T stack. Use the helper first:

```bash
./scripts/install_training_deps_jetson.sh
```

If `torchvision` still fails to import, install the NVIDIA/Jetson-compatible `torchvision` build for the PyTorch version on the device, then rerun the helper.

Train from all reports:

```bash
robot-vision-train-vision \
  --reports data/reports \
  --output data/models/pass_fail_classifier \
  --epochs 5 \
  --batch-size 8
```

Train only one recipe:

```bash
robot-vision-train-vision --recipe 1inch
```

The trainer saves a Hugging Face model, image processor, and `robot_vision_training_manifest.json` under the output folder. Training requires both PASS and FAIL examples.

## Current Limits

- The real Astra Plus Pro SDK adapter still needs to be wired and validated on the Jetson.
- Dimension accuracy depends on calibration plate quality, camera mounting stability, and lens/depth alignment.
- PLC/GPIO/ROS2 topic triggers are not active yet; the first version uses manual snap and image-based auto presence detection.
