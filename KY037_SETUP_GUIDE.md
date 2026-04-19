# KY-037 Sound Sensor Setup Guide for Raspberry Pi

This guide is for a KY-037 sound sensor module connected directly to a
Raspberry Pi digital GPIO pin.

The reference paper `Design_of_a_Classroom_Noise_Monitoring_Tool_Using_.pdf`
uses the KY-037 as part of a classroom noise monitor and describes behavior
around classroom thresholds such as 45 dB and 55 dB. This direct Raspberry Pi
setup uses the KY-037 digital output only, so it can report whether sound is
above or below the module threshold. It cannot measure calibrated dB without an
ADC connected to the analog output.

## Important Notes

- Use `3.3V` when connecting the KY-037 digital output directly to a Raspberry
  Pi GPIO pin.
- Do not connect KY-037 `AO` / `A0` directly to the Raspberry Pi. Raspberry Pi
  GPIO pins are digital-only and cannot read analog voltage.
- Do not power the KY-037 from `5V` when `DO` / `D0` is connected directly to
  the Raspberry Pi, because a 5V digital output can damage the GPIO pin.
- This test is a threshold monitor, not a calibrated sound level meter.

## Wiring

Power off the Raspberry Pi before wiring.

| KY-037 pin | Raspberry Pi pin | GPIO | Purpose |
| --- | --- | --- | --- |
| `VCC` or `+` | Pin `1` | `3.3V` | Power |
| `GND` or `G` | Pin `6` | `GND` | Ground |
| `DO` or `D0` | Pin `7` | `GPIO4` | Digital threshold output |
| `AO` or `A0` | Leave unconnected | N/A | Analog output, requires ADC |

## Install Prerequisite

```bash
sudo apt update
sudo apt install python3-rpi.gpio
```

## Test Script

The test script is:

- `tests/tests_on_pi/ky037_sound_threshold_test.py`

Run the continuous sound detector:

```bash
cd ~/AISENTINEL
python3 tests/tests_on_pi/ky037_sound_threshold_test.py
```

The script prints only:

```text
no sound detected
sound detected
```

This version follows the referenced Raspberry Pi tutorial pattern: `GPIO4`
reads `1` as `sound detected` and `0` as `no sound detected`.

## Calibration

1. Wire the KY-037 as shown above and boot the Raspberry Pi.
2. Start the script.
3. Keep the classroom quiet and watch the terminal output.
4. Turn the KY-037 potentiometer slowly until normal room sound stays mostly
   `no sound detected`.
5. Speak loudly, clap, or create the classroom sound level you want to flag.
6. Continue adjusting until the onboard KY-037 digital LED and the script switch
   to `sound detected` at the desired warning point.

For a PDF-style classroom monitor, use the 45 dB and 55 dB values as behavior
targets when calibrating against a real sound meter or phone SPL app. With this
digital-only setup, the Pi records only threshold crossings.

## Troubleshooting

- Always `sound detected`:
  - Recheck `DO` / `D0` to GPIO4 physical pin `7`.
  - Adjust the potentiometer away from the most sensitive position.
- Always `no sound detected`:
  - Try speaking or clapping close to the microphone.
  - Recheck `VCC` to `3.3V` and `GND` to ground.
  - Adjust the potentiometer until the KY-037 onboard digital LED toggles.
- GPIO permission or import error:
  - Install GPIO support with `sudo apt install python3-rpi.gpio`.
  - Run on Raspberry Pi OS, not on a desktop test machine.
- Need real dB readings:
  - Add an ADC such as MCP3008 or ADS1115 and read KY-037 `AO` / `A0` through
    the ADC.
  - Calibrate against a known sound level meter before using 45 dB or 55 dB as
    numeric thresholds.
