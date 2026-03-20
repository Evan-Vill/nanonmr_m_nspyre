# Project Guidelines

## Code Style
Follow PEP 8 for Python code. Use type hints where possible. Prefer dataclasses for configuration objects (see `nv_dataclasses_2026_03_05.py` for examples). Avoid scattering magic numbers; centralize defaults in `experiment_defaults.py`.

## Architecture
This is an nspyre-based quantum measurement system for NV-center NMR experiments. The system orchestrates multi-instrument control (RF generators, AWGs, digitizers, stage control) with real-time data streaming and live plotting.

**Core Stack:**
- **Framework**: nspyre 0.6+ — experiment orchestration, data server, RPC gateway
- **UI**: PyQt6 with pyqtgraph for real-time plots
- **Hardware**: Zurich HDAWG, Rohde-Schwarz SG396, NI-DAQ, Spectrum digitizer, PulseStreamer
- **Data Flow**: Instruments → nspyre DataServer → GUI subscriptions → Live plots

**Multi-Process Architecture:**

The system uses three decoupled processes that communicate via RPC and a shared data server:

```
┌─────────────────────────────────────────────────────────────────┐
│ GUI Process (app.py) — PyQt6 UI                                 │
│ ├─ Instruments Widget: Monitors hardware status (status_queue)   │
│ ├─ Experiments Widget: Submits experiments via queue_to_exp      │
│ ├─ Plots Widget: Subscribes to live data from DataServer        │
│ └─ Calculations Widget: Processes/displays results              │
└──────────────────┬────────────────────────────────┬─────────────┘
                   │ RPC Calls (port 42068)         │ DataSource
                   │ (InstrumentManager)            │ subscriptions
                   ↓                                ↓
        ┌──────────────────────────┐    ┌──────────────────────┐
        │ Instrument Server        │    │ nspyre DataServer    │
        │ (my_inserv.py,           │    │ (dataserv process)   │
        │  remote_inserv.py)       │    │                      │
        │                          │    │ - Streaming buffer   │
        │ Device Drivers:          │    │ - Subscription mgmt  │
        │ ├─ SG396 (RF gen)        │    │ - Real-time updates  │
        │ ├─ HDAWG (AWG)           │    │ - Dataset persistence│
        │ ├─ PulseStreamer         │    └──────────────────────┘
        │ ├─ Spectrum Digitizer    │
        │ ├─ NI-DAQ                │
        │ └─ Stage, Laser, etc.    │
        └──────────────────────────┘
                   ↓
            ┌──────────────┐
            │ Physical     │
            │ Hardware     │
            └──────────────┘
```

**1. Instrument Server (Port 42068 RPC)**
- Runs in separate process (`my_inserv.py`, `remote_inserv.py`)
- Instantiates all hardware drivers and wraps them as RPC-callable objects
- Serves as single proxy to physical equipment — prevents concurrent hardware access conflicts
- Example: `mgr.sg.set_frequency(2.87e9)` sends RPC call across process boundary to driver instance
- `InstrumentManager` context manager (in app.py or experiments) auto-connects to these RPC proxies
- Drivers defined in `src/experiments/drivers/` (e.g., `sg396_driver.py`, `hdawg_driver.py`)

**2. nspyre DataServer (Default Port 5555)**
- Runs as independent process (`nspyre-dataserv` shell command)
- Central buffer/hub for experiment results
- Experiments push results via `data.push({'datasets': {...}, 'title': '...', ...})`
- GUI subscribes to updates: `DataSource(dataset_name)` auto-receives new data on push
- Real-time plotting: pyqtgraph plots trigger on each push (no polling)
- Persists datasets to disk (RPC gateway for cross-process streaming)
- **Critical**: Must be running before GUI starts, or GUI cannot subscribe to plots

**3. Experiment Logic & Orchestration**
- Experiments are methods decorated with `@managed_experiment`
- Decorator provides three key objects:
  - `mgr` (InstrumentManager): RPC proxy context to instrument server
  - `data` (DataSource): Stream-to-GUI channel for real-time plots
  - `token`: Unique experiment run ID (used for exclusive PulseStreamer control)
- Experiment runs in subprocess (spawned by GUI's experiments widget)
- Communication back to GUI: status updates via `queue_from_exp` (progress %, fit values, etc.)
- Subscriber pattern: GUI can request stop via `queue_to_exp`
- On exception or completion, decorator's finally block releases PulseStreamer token & closes experiment context

**Data Flow Example (ODMR Scan):**
```
User clicks "Run ODMR" in GUI
  ↓ [queue_to_exp]
Experiment subprocess spawns nv_experiments_2026_03_18.odmr_scan()
  ├─ Acquires InstrumentManager context (RPC connects to my_inserv)
  ├─ Loop over frequencies:
  │  ├─ RPC: mgr.sg.set_frequency(f) → SG396 driver → RF generator
  │  ├─ RPC: mgr.hdawg.play_pulse_sequence() → HDAWG driver → AWG
  │  ├─ RPC: mgr.dig.read() → Digitizer driver → data acquisition
  │  ├─ Process data (reshape, average, fit)
  │  ├─ data.push({'datasets': {'odmr': result}, ...}) 
  │  │   → DataServer → GUI subscription updates plot in real-time
  │  └─ queue_from_exp.put(status_msg) → GUI progress bar
  └─ Finally: release PulseStreamer token, close contexts

Experiment completes → result saved via customUtils.run_save()
```

All hardware accessed through RPC proxies via InstrumentManager. Experiments use `@managed_experiment` decorator for clean lifecycle management.

## Build and Test
**Installation:**
```bash
pip install -e .
```

**Running the system** (in order, separate terminals):
```bash
# Terminal 1: Local instrument server (RPC gateway to hardware drivers)
python src/experiments/drivers/my_inserv.py
# Listens on localhost:42068; wraps SG396, HDAWG, digitizer, etc. as RPC objects

# Terminal 2: Remote instrument server (empty placeholder for distributed systems)
python src/experiments/drivers/remote_inserv.py
# For future expansion; currently shadows local_inserv on same port

# Terminal 3: nspyre dataserver (Real-time data streaming to GUI)
nspyre-dataserv
# Listens on localhost:5555; buffers experiment results, manages subscriptions

# Terminal 4: GUI application (User-facing control & plots)
python src/experiments/app.py
# Connects to both InstrumentManager (port 42068) and DataServer (port 5555)
```

**Startup Order Matters:**
- If DataServer starts after GUI: GUI cannot subscribe to plots (silent failure)
- If Instrument Server missing: GUI blocks on `InstrumentManager()` (hangs indefinitely)
- Common fix: Kill all Python processes, restart in order

No formal test suite exists — project uses interactive GUI testing with manual validation.

## Conventions
- **File Naming by Date**: The most recent file is "active development" (e.g., `nv_experiments_2026_03_18.py`). Always check timestamps — old files may be used as templates but shouldn't be imported.
- **Parameter Defaults Centralized**: All default values in `experiment_defaults.py`.
- **Subsequence Count Dictionary**: Use `SUBSEQ_COUNT` for correct data reshaping in experiments.
- **Sideband Selection**: Follow the pattern in `choose_sideband()` for RF modulation.
- **Data Streaming**: Stream results progressively via `data.push()` for real-time GUI updates.

**Common Pitfalls:**
- Forgot to start all servers: System hangs silently on first `InstrumentManager()` call.
- Port conflicts: Kill lingering Python processes if port 42068 is in use.
- Hardware verification: Test digitizer connection before running experiments.
- Exclusive control: PulseStreamer tokens must be released on crash.</content>
<parameter name="filePath">c:\Users\nanon\nanon-nspyre\nanonmr_m_nspyre\.github\copilot-instructions.md