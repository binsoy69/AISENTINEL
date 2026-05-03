# KY-037 + ADS1015 Setup Guide for Raspberry Pi

The reference paper `Design_of_a_Classroom_Noise_Monitoring_Tool_Using_.pdf`
uses the KY-037 as part of a classroom noise monitor and describes behavior
around classroom thresholds such as 45 dB and 55 dB. This setup uses the
KY-037 analog output through an ADS1015 ADC so the Raspberry Pi can calibrate
the sensor once and then continuously output estimated dB. The calibrated
sensor is now integrated into the front-node runtime and the central dashboard
stack, so the front node can report live classroom noise during an active
monitoring session.

## Important Notes

- Use `3.3V` for both the KY-037 and the ADS1015 when wiring directly to a
  Raspberry Pi.
- Do not connect KY-037 `AO` / `A0` directly to the Raspberry Pi. Raspberry Pi
  GPIO pins are digital-only and cannot read analog voltage.
- This setup is suitable for calibrated classroom noise categories, but the
  KY-037 is still not a certified SPL meter. Treat the output as estimated dB
  after calibration.

## Wiring

Power off the Raspberry Pi before wiring.

### KY-037 to ADS1015

| KY-037 pin | ADS1015 pin | Purpose |
| --- | --- | --- |
| `VCC` or `+` | Shared Pi `3.3V` rail | Power |
| `GND` or `G` | Shared Pi `GND` rail | Ground |
| `AO` or `A0` | `A0` | Analog classroom-noise signal |
| `DO` or `D0` | Leave unconnected | Optional digital threshold output, not used by this script |

### ADS1015 to Raspberry Pi

| ADS1015 pin | Raspberry Pi pin | GPIO | Purpose |
| --- | --- | --- | --- |
| `VDD` | Pin `1` | `3.3V` | Power |
| `GND` | Pin `6` | `GND` | Ground |
| `SDA` | Pin `3` | `GPIO2` | I2C data |
| `SCL` | Pin `5` | `GPIO3` | I2C clock |
| `ADDR` | Pin `6` | `GND` | Select default I2C address `0x48` |
| `A0` | From KY-037 `AO` | N/A | Analog input channel used by the script |
| `ALERT/RDY` | Leave unconnected | N/A | Not used by the test script |

## Install Prerequisite

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable

sudo apt update
sudo apt install python3-smbus i2c-tools

# Confirm the ADS1015 is visible on I2C bus 1
i2cdetect -y 1
```

If `ADDR` is tied to `GND`, `i2cdetect -y 1` should normally show `48`.

## Scripts

This setup now uses no-argument launchers plus one shared runtime module:

- shared runtime helpers: `runtime/edge_node_runtime/sound_monitor.py`
- front-node calibration launcher: `programs/calibrate_front_sound_sensor.py`
- mid-node calibration launcher: `programs/calibrate_mid_sound_sensor.py`
- advanced calibration script: `runtime/central_dashboard/scripts/calibrate_sound_sensor.py`
- raw ADS monitor: `programs/test_sound_sensor_raw.py`
- test script: `tests/tests_on_pi/diagnostics/ky037_sound_threshold_test.py`
- default front config file:
  `runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json`
- default mid config file:
  `runtime/central_dashboard/data/node_mid/sound/ky037_ads1015_config.json`

The calibration and test scripts use the ADS1015 sampling and dB estimation
logic from `runtime/edge_node_runtime/sound_monitor.py`, so the runtime and
setup tools use the same implementation.

## Runtime Integration

When sound monitoring is enabled in the front-node runtime config:

- the front node samples the KY-037 only while a monitoring session is active
- the front-node local dashboard shows live estimated dB, threshold, and
  sensor state
- the central shared dashboard shows the front-node noise telemetry through the
  node heartbeat
- a `Noise Threshold Exceeded` incident is saved only when the estimated dB
  crosses above the configured threshold
- repeated loud windows are suppressed until the level drops below the
  threshold and the cooldown has elapsed

Noise incidents save one JPG snapshot from the latest front-node camera frame.
They intentionally do not create GIF evidence; GIFs are reserved for visual
student-behavior incidents.

If the front node is connected to the shared central dashboard, also review
the network settings in `runtime/central_dashboard/README.md`. The sound sensor
does not add new host-IP fields. The node still uses `node_front.ini` for
`host`, `port`, and `central_base_url`, and the central service still uses
`central_service.ini` for its `host` and `port`.

## Runtime Config

Add or update the `[sound_sensor]` section in the front-node runtime INI:

```ini
[sound_sensor]
enabled = true
calibration_config = runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json
alert_threshold_db = 55.0
incident_cooldown_sec = 10.0
i2c_bus = 1
i2c_address = 0x48
adc_channel = 0
full_scale = 4.096
data_rate = 1600
sample_interval = 0.002
window_seconds = 1.0
```

Field meaning:

- `enabled`: turns session-scoped sound monitoring on or off
- `calibration_config`: saved JSON file produced by the calibration script
- `alert_threshold_db`: dB value that triggers the live dashboard warning and
  saved noise incident
- `incident_cooldown_sec`: minimum time between saved threshold-crossing noise
  incidents
- `i2c_bus`, `i2c_address`, `adc_channel`, `full_scale`, `data_rate`,
  `sample_interval`, `window_seconds`: ADS1015 sampling settings used at
  runtime

Recommended repo defaults:

- enable the sensor only on the real front-node webcam configs
- keep it disabled on video playback configs and on the mid node

## Calibration Script

The calibration script saves the sensor settings and the two reference points
used for estimated dB mapping.

Run the full interactive calibration from the repository root on the Pi that
has the KY-037 connected:

```bash
cd ~/AISENTINEL
python3 programs/calibrate_front_sound_sensor.py
```

It will guide you through:

1. capturing the quiet reference near `45 dB`
2. capturing the loud reference near `55 dB`

The launcher automatically:

- saves the JSON to
  `runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json`
- updates `config/front_node.ini` `[sound_sensor] calibration_config`
- sets `[sound_sensor] enabled = true` after both references are present

Use the mid-node launcher only if the sound sensor is attached to the mid Pi:

```bash
python3 programs/calibrate_mid_sound_sensor.py
```

You can also capture each point separately:

```bash
python3 runtime/central_dashboard/scripts/calibrate_sound_sensor.py --config config/front_node.ini --capture-quiet
python3 runtime/central_dashboard/scripts/calibrate_sound_sensor.py --config config/front_node.ini --capture-loud
```

To save only the runtime settings without capturing references:

```bash
python3 runtime/central_dashboard/scripts/calibrate_sound_sensor.py \
  --config config/front_node.ini \
  --quiet-db 45 \
  --loud-db 55 \
  --window-seconds 1.0 \
  --data-rate 1600 \
  --save-config
```

To inspect the saved values:

```bash
python3 runtime/central_dashboard/scripts/calibrate_sound_sensor.py --config config/front_node.ini --show-config
```

Example saved config file:

```json
{
  "bus": 1,
  "address": "0x48",
  "channel": 0,
  "full_scale": 4.096,
  "data_rate": 1600,
  "sample_interval": 0.002,
  "window_seconds": 1.0,
  "quiet_db": 45.0,
  "loud_db": 55.0,
  "ref_quiet_rms_mv": 14.12,
  "ref_loud_rms_mv": 29.84,
  "updated_at": "2026-04-22T00:00:00Z"
}
```

## Raw ADS Monitor

Use this first when you want to confirm that the KY-037 analog output is
changing through ADS1015 channel `A0`, without completing calibration:

```bash
python3 programs/test_sound_sensor_raw.py
```

It reads the ADS settings from `config/front_node.ini` and prints continuously
until Ctrl+C:

```text
KY-037 ADS1015 Raw Monitor
address=0x48 channel=A0 full_scale=+/-4.096V rate=1600SPS read_interval=0.1s
Press Ctrl+C to stop.
sample=1 raw_code=812 voltage=1.6240V voltage_mv=1624.00
sample=2 raw_code=817 voltage=1.6340V voltage_mv=1634.00
```

Advanced usage can run the diagnostic script directly and choose a channel or
print a fixed number of reads:

```bash
python3 tests/tests_on_pi/diagnostics/ky037_ads1015_raw_test.py \
  --channel 0 \
  --read-interval 0.05 \
  --count 20
```

## Test Script

After calibration, run the estimated dB monitor:

```bash
python3 programs/test_sound_sensor.py
```

Default output looks like:

```text
estimated_db=47.3 status=warning
```

Debug mode adds RMS and voltage details:

```bash
python3 tests/tests_on_pi/diagnostics/ky037_sound_threshold_test.py \
  --config-file runtime/central_dashboard/data/node_front/sound/ky037_ads1015_config.json \
  --debug
```

Example debug output:

```text
estimated_db=47.3 status=warning rms=18.65mV p2p=95.00mV
debug address=0x48 channel=A0 mean=1.6320V min=1.5900V max=1.6740V rate=1600SPS samples=480
```

The test script expects the calibration file to already contain both reference
values. If they are missing, it will stop and tell you to run the calibration
script first.

## Dashboard Behavior

- Below threshold: dashboards show the current dB and normal sensor state.
- At or above threshold: dashboards show a live noise alert banner/status.
- On threshold crossing: a `Noise Threshold Exceeded` incident is saved.
- While the level stays above threshold: no duplicate incidents are created.
- After the level drops below threshold and rises again: a new incident can be
  created after cooldown.

## Calibration Notes

1. Wire the KY-037 and ADS1015 as shown above and boot the Raspberry Pi.
2. Use a sound level meter or phone SPL app at the same location as the KY-037.
3. During quiet capture, hold the room near the paper's lower threshold,
   typically `45 dB`.
4. During loud capture, hold the room near the paper's upper threshold,
   typically `55 dB`.
5. The saved config then lets the test script classify windows like the paper:
   - below `45 dB`: `status=normal`
   - `45 dB` up to below `55 dB`: `status=warning`
   - `55 dB` and above: `status=loud`

## Troubleshooting

- The central service or node agent says `Cannot assign requested address`:
  - This usually means `host` was set to an IP address that does not belong to
    the machine running that process.
  - Keep `host = 0.0.0.0` in `runtime/central_dashboard/central_service.ini`
    and in `runtime/central_dashboard/node_front.ini` unless you intentionally
    want to bind to one specific local interface.
  - Set `central_base_url` in `runtime/central_dashboard/node_front.ini` to the
    actual IP of the machine running the central dashboard, for example
    `http://192.168.1.50:8090`.
  - Use `hostname -I` on Raspberry Pi OS or `ipconfig` on Windows to confirm
    the real machine IP first.
- `i2cdetect -y 1` does not show `48`:
  - Recheck `SDA` to Pi pin `3` and `SCL` to Pi pin `5`.
  - Confirm I2C is enabled in `raspi-config`.
  - Recheck `ADDR` wiring. `ADDR -> GND` gives address `0x48`.
- The test script says calibration is incomplete:
  - Run `python3 programs/calibrate_front_sound_sensor.py`.
  - Or run `--capture-quiet`, then `--capture-loud`.
  - Use `--show-config` to confirm the values were written to the JSON file.
- RMS and peak-to-peak values do not change with sound:
  - Make sure KY-037 `AO` / `A0` is connected to ADS1015 `A0`.
  - Recheck KY-037 `VCC` and `GND`.
  - Turn the KY-037 potentiometer and watch the `rms=...mV` output again.
- The script says it cannot communicate with ADS1015:
  - Install I2C support with `sudo apt install python3-smbus i2c-tools`.
  - Run on Raspberry Pi OS, not on a desktop test machine.
- Estimated dB values look wrong:
  - Re-run the calibration script with an external meter.
  - Keep the sensor and reference meter in the same position during calibration.
