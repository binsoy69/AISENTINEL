# KY-037 + ADS1015 Setup Guide for Raspberry Pi

The reference paper `Design_of_a_Classroom_Noise_Monitoring_Tool_Using_.pdf`
uses the KY-037 as part of a classroom noise monitor and describes behavior
around classroom thresholds such as 45 dB and 55 dB. This updated setup uses
the KY-037 analog output through an ADS1015 ADC so the Raspberry Pi can measure
an analog classroom-noise signal and map it to paper-style categories after
calibration.

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

## Test Script

The test script is:

- `tests/tests_on_pi/ky037_sound_threshold_test.py`

Run the uncalibrated analog monitor:

```bash
cd ~/AISENTINEL
python3 tests/tests_on_pi/ky037_sound_threshold_test.py
```

Uncalibrated output looks like:

```text
status=uncalibrated rms=12.84mV p2p=74.00mV
```

Show raw debug information for calibration:

```bash
python3 tests/tests_on_pi/ky037_sound_threshold_test.py --debug
```

Example debug line:

```text
debug address=0x48 channel=A0 mean=1.6320V min=1.5900V max=1.6740V rate=1600SPS samples=480
```

Run with paper-style dB mapping after calibration:

```bash
python3 tests/tests_on_pi/ky037_sound_threshold_test.py --ref-quiet-rms-mv 14 --ref-loud-rms-mv 30
```

Calibrated output looks like:

```text
estimated_db=47.3 status=warning rms=18.65mV p2p=95.00mV
```

## Calibration

1. Wire the KY-037 and ADS1015 as shown above and boot the Raspberry Pi.
2. Start the script.
3. Use a sound level meter or phone SPL app at the same location as the KY-037.
4. Measure a quiet classroom condition near the paper's lower threshold and note
   the script's `rms=...mV` value when the external meter is about `45 dB`.
5. Measure a louder classroom condition and note the script's `rms=...mV` value
   when the external meter is about `55 dB`.
6. Rerun the script with those two RMS values:

```bash
python3 tests/tests_on_pi/ky037_sound_threshold_test.py \
  --ref-quiet-rms-mv 14 \
  --ref-loud-rms-mv 30
```

7. The script will then classify windows like the paper:
   - below `45 dB`: `status=normal`
   - `45 dB` up to below `55 dB`: `status=warning`
   - `55 dB` and above: `status=loud`

## Troubleshooting

- `i2cdetect -y 1` does not show `48`:
  - Recheck `SDA` to Pi pin `3` and `SCL` to Pi pin `5`.
  - Confirm I2C is enabled in `raspi-config`.
  - Recheck `ADDR` wiring. `ADDR -> GND` gives address `0x48`.
- The script only reports `status=uncalibrated`:
  - This is expected until you pass both `--ref-quiet-rms-mv` and
    `--ref-loud-rms-mv`.
- RMS and peak-to-peak values do not change with sound:
  - Make sure KY-037 `AO` / `A0` is connected to ADS1015 `A0`.
  - Recheck KY-037 `VCC` and `GND`.
  - Turn the KY-037 potentiometer and watch the `rms=...mV` output again.
- The script says it cannot communicate with ADS1015:
  - Install I2C support with `sudo apt install python3-smbus i2c-tools`.
  - Run on Raspberry Pi OS, not on a desktop test machine.
- Estimated dB values look wrong:
  - Recalibrate `--ref-quiet-rms-mv` and `--ref-loud-rms-mv` with an external
    meter.
  - Keep the sensor and reference meter in the same position during calibration.
