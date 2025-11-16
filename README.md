<h1 align="center">SRM Traffic Violation Detection System</h1>
<p align="center"><strong>Final Year Project - SRM Institute of Science & Technology</strong></p>
<p align="center">AI-powered monitoring that detects, analyzes, and reports real-time traffic violations using computer vision, OCR, and predictive analytics.</p>

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture & Design](#architecture--design)
4. [Quick Start](#quick-start)
5. [Requirements & Installation](#requirements--installation)
6. [Usage Guide](#usage-guide)
7. [Detection Modules](#detection-modules)
8. [Dashboard & Analytics](#dashboard--analytics)
9. [Output Gallery](#output-gallery)
10. [Troubleshooting & Performance](#troubleshooting--performance)
11. [Technical Details](#technical-details)
12. [Support & Contribution](#support--contribution)
13. [License](#license)
14. [Credits](#credits)

## Overview
The SRM Traffic Violation Detection System is an end-to-end platform for monitoring urban intersections. It ingests recorded or live CCTV feeds, detects rule violations using YOLOv8 models, reads license plates with OCR, correlates events with vehicle speed and behavior, and produces dashboards, alerts, and compliance reports for enforcement teams.

## Features
### Core Detection
- Vehicle detection and classification with YOLOv8
- License plate recognition with OCR and language-aware parsing
- Real-time speed estimation with trajectory tracking
- Helmet compliance checks for riders and passengers
- Red-light signal monitoring with stop-line awareness
- Lane discipline detection for wrong-way and unsafe changes

### Advanced Platform Capabilities
- Behavioral analytics that flag aggressive maneuvers
- Predictive heatmaps for violation hotspots
- Instant alerts via dashboard widgets and email
- Rich dashboard with KPIs, charts, and drill-down history
- Automated report generation for evidence and audits

## Architecture & Design
- **System Architecture** - Data ingestion, AI inference, PostgreSQL storage, and alerting pipeline.

  ![Architecture Diagram](./assets/output_screenshots/image2.png)

- **Workflow** - Upload/stream -> frame sampling -> detection -> violation logging -> visualization.

  ![Workflow Diagram](./assets/output_screenshots/image1.png)

- **Class Diagram** - Core services covering detectors, repositories, and UI components.

  ![Class Diagram](./assets/output_screenshots/image3.png)

- **Sequence Diagram** - Event timeline from video ingestion to alert notification.

  ![Sequence Diagram](./assets/output_screenshots/image4.png)

## Quick Start
> Need a fast demo? Use the packaged runner; the Streamlit UI boots automatically.

```bash
# Recommended path
python run_system.py
```

```bash
# Manual launch
streamlit run app.py
```

**Default credentials:** `admin` / `admin123`

## Requirements & Installation
### System Requirements
- Python 3.8+
- 4 GB RAM minimum (8 GB recommended)
- ~2 GB free storage
- Windows, macOS, or Linux

### Python Dependencies
```bash
pip install streamlit numpy pandas opencv-python psycopg2-binary bcrypt
```

Optional deep-learning extras:
```bash
pip install ultralytics easyocr torch torchvision
```

### Installation Steps
1. Clone or download the repository and move into the project root.
2. Create and activate a virtual environment (recommended).
3. `pip install -r requirements.txt`
4. `python run_system.py`
5. Open `http://localhost:8501` in a browser and log in.

### Optional Database Setup
- Update connection details in `database.py` for PostgreSQL.
- Apply schemas from `database_scripts/`.
- When unreachable, the system automatically falls back to demo (in-memory) mode.

## Usage Guide
1. **Video Upload & Processing** - Select `Video Upload & Processing`, drop MP4/AVI/MOV/MKV/WebM files, configure confidence threshold, frame skip, and violation types, then click **Start Processing**.
2. **Live Status** - Monitor processing metrics, current frame, and event counters in real time.
3. **Violation Review** - Switch to the Violations tab to inspect evidence snapshots, license plates, timestamps, and severity tags.
4. **Analytics & Reports** - Explore dashboards, download CSV/PDF reports, and export violation bundles.
5. **Alerts** - Configure SMTP settings, recipients, and rule thresholds in the Alerts module to broadcast incidents.

## Detection Modules
### Speeding
- Multi-point trajectory plus perspective calibration
- Configurable road-wise limits and severity bands

### Helmet Compliance
- Detects riders and passengers on motorcycles, scooters, or bicycles
- Flags missing helmets even in dense traffic

### Red-Light Monitoring
- Tracks traffic signal state vs. vehicle position
- Detects stop-line intrusions and start delays

### Lane Discipline
- Identifies wrong-way driving, unsafe lane changes, and solid-line crossings

### Additional Behaviors
- Aggressive acceleration/braking cues
- Tailgating and erratic maneuvers (beta)

## Dashboard & Analytics
- Overview metrics (totals, per-vehicle breakdown, top violations)
- Historical trends with daily/weekly filters
- Violation heatmap pinpointing hotspots
- Vehicle type distribution and compliance scorecards
- Processed video playback with overlays for audit trails

## Output Gallery
| Admin Login | Video Upload | Upload Detail |
| --- | --- | --- |
| ![Admin login screen](./assets/output_screenshots/image5.png) | ![Video upload screen](./assets/output_screenshots/image6.png) | ![Upload video card](./assets/output_screenshots/image7.png) |

| Processing | Violations List | Violation History |
| --- | --- | --- |
| ![Processing status](./assets/output_screenshots/image8.png) | ![Detected violations list](./assets/output_screenshots/image9.png) | ![Violation history dashboard](./assets/output_screenshots/image10.png) |

| Statistics Dashboard | Violation Dashboard | Advanced Analytics |
| --- | --- | --- |
| ![Statistics dashboard](./assets/output_screenshots/image11.png) | ![Traffic violation dashboard](./assets/output_screenshots/image12.png) | ![Advanced AI analytics](./assets/output_screenshots/image13.png) |

| Predictive Analytics | Reports | Live Alerts |
| --- | --- | --- |
| ![Predictive analytics](./assets/output_screenshots/image14.png) | ![Reports view](./assets/output_screenshots/image15.png) | ![Live alerts monitoring](./assets/output_screenshots/image16.png) |

| Alert Configuration | Sample Alert Email | |
| --- | --- | --- |
| ![Alert configuration view](./assets/output_screenshots/image17.png) | ![Sample alert email](./assets/output_screenshots/image18.png) | |

## Troubleshooting & Performance
**No violations detected**
- Verify source video quality and lighting.
- Lower confidence threshold (0.3-0.5) and disable frame skipping for testing.

**YOLO/OCR not available**
- Install optional dependencies (`ultralytics`, `easyocr`, `torch`, `torchvision`).
- Ensure your GPU/CPU meets Torch requirements.

**Database connection failed**
- Confirm credentials and network reachability; system will still operate in demo mode.

**Video processing feels slow**
- Increase frame skip to 5-10, trim video length, and close other heavy apps.

**Performance tuning tips**
- Frame skip increase -> faster but less granular
- Confidence increase -> fewer false positives
- Lower video resolution -> faster inference
- Process long recordings in batches

## Technical Details
- **Models** - YOLOv8 for vehicles, EasyOCR for license plates, custom CV pipelines for speed and lane detection.
- **Algorithms** - Multi-point trajectory analysis, helmet texture heuristics, color phase detection for signals, lane boundary tracking.
- **Accuracy** - 85-95% vehicle detection, 70-90% license plate reading, +/-5 km/h speed tolerance, 80-95% violation classification.
- **Codebase** - Streamlit front end, Python back end, modular detectors defined in `detection_models.py`, orchestration handled by `video_processor.py` and `run_system.py`.

## Support & Contribution
- Run `python test_violations.py` for diagnostics.
- Review terminal logs for stack traces or OCR warnings.
- For contributions: fork -> create a feature branch -> implement changes -> run tests -> open a pull request with screenshots/logs.

## License
Educational use only for SRM Institute of Science & Technology projects. Contact the maintainers before deploying commercially.

## Credits
**Project Done By CHANDRAMOULI GM**
**SRM STUDENT**
**SRM Institute of Science & Technology**
**Final Year Project - Traffic Violation Detection System**

