# INMP441 Microphone Setup Guide for Raspberry Pi

This guide is for the common `INMP441` I2S microphone breakout on the 40-pin Raspberry Pi header.

Assumptions:
- Raspberry Pi 5 with the standard 40-pin header
- Raspberry Pi OS Bookworm or Trixie
- One `INMP441` microphone module

## Important notes

- The `INMP441` is a **digital I2S microphone**. Do not connect it like an analog mic.
- Use **3.3V only**. Do **not** connect the module to Raspberry Pi `5V`.
- The Raspberry Pi input pin for a single I2S microphone is the data-in line on `GPIO20`.

## Wiring

Connect the module like this:

| INMP441 pin | Raspberry Pi pin | GPIO | Purpose |
| --- | --- | --- | --- |
| `VDD` | Pin `1` | `3.3V` | Power |
| `GND` | Pin `6` | `GND` | Ground |
| `SCK` or `BCLK` | Pin `12` | `GPIO18` | I2S bit clock |
| `WS` or `LRCL` | Pin `35` | `GPIO19` | I2S word select |
| `SD` or `DOUT` | Pin `38` | `GPIO20` | I2S data into Raspberry Pi |
| `L/R` | Pin `6` to select left, or Pin `1` to select right | `GND` or `3.3V` | Channel select |

Optional:

- If your board exposes `CHIPEN`, connect it to `3.3V` so the mic stays enabled.
- `GPIO21` on physical Pin `40` is the Pi's I2S data-out line. You do not use it for one `INMP441` input mic.

## Quick physical checklist

1. Power off the Raspberry Pi.
2. Wire `VDD`, `GND`, `SCK`, `WS`, and `SD` exactly as above.
3. Tie `L/R` to `GND` if you want the mic on the left channel.
4. Boot the Pi.
5. Confirm ALSA can see a capture device with:

```bash
arecord -l
```

If no capture device appears, the mic is not yet exposed correctly to Linux. That is usually a wiring issue or an I2S device-tree / overlay issue.

## Test script in this repo

I added a test script here:

- `tests/tests_on_pi/inmp441_mic_test.py`

Run it like this:

```bash
python3 tests/tests_on_pi/inmp441_mic_test.py --list
python3 tests/tests_on_pi/inmp441_mic_test.py --device plughw:1,0
```

The script:

- Lists ALSA capture devices
- Shows the current dBFS level live while recording
- Records a WAV sample
- Prints peak and RMS levels
- Flags silence, low signal, or clipping

## Install prerequisite

```bash
sudo apt update
sudo apt install alsa-utils
```

## Troubleshooting

- No capture device in `arecord -l`:
  Your I2S mic is not configured at the OS level yet, or the wiring is wrong.
- File records but level stays near zero:
  Recheck `SD -> GPIO20`, `WS -> GPIO19`, and `SCK -> GPIO18`.
- Very noisy or unstable reading:
  Recheck `GND`, keep wires short, and make sure the module is powered from `3.3V`.
- Captured signal is only on one channel:
  That is normal for a single `INMP441`. The `L/R` pin selects whether it lands on left or right.

## Reference notes

According to Raspberry Pi's official I2S documentation for Raspberry Pi 5, the relevant pins are:

- `GPIO18` = `I2S0_SCLK`
- `GPIO19` = `I2S0_WS`
- `GPIO20` = `I2S0_SDI[0]`
- `GPIO21` = `I2S0_SDO[0]`

According to the official `INMP441` datasheet:

- `SCK` is the serial data clock input
- `WS` is the word-select input
- `SD` is the serial data output
- `L/R` low = left channel, high = right channel
