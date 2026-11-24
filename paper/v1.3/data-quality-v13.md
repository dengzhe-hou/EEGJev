# EEGJev-Bench v1.3 data quality audit

This page records the signal rules and aggregate checks described in Supplementary Note 2 of the manuscript at revision dd6f1cf. The checks use the original releases as references.

## Signal pipeline

- Each recording is read with its source's fixed channel list and order. Recordings with missing or mismatched channels are excluded. Channels are never zero-filled.
- A mains component at 50, 60 or 100 Hz is removed with a zero-phase notch filter over the whole recording if the median power ratio to frequencies 3 to 8 Hz away is at least 3 in sampled training recordings. This is the only filter.
- The whole recording is resampled to 200 Hz before cutting windows at their recording times. Windows that start in the first 2 s, end in the last 0.5 s, or lie within 0.1 s of a discontinuity marker are excluded.
- Exclude a window that overlaps at least 0.1 s with all channels constant, or has one channel constant throughout. Standardize each retained window per channel.
- ERP CORE alone is re-referenced, by subtracting the mean of its 30 scalp channels. [ERP CORE data descriptor](https://doi.org/10.1016/j.neuroimage.2020.117465)
- HBN uses 120 channels from E1–E128, after excluding E30, E47, E48, E49, E67, E113, E119 and E126. These eight channels were constant in at least 5% of the windows checked in the channel audit. Cz is excluded from the E1–E128 channel list. COG-BCI uses 62 channels, omitting Cz. LEMON uses the 17 channels present in all its files. The depression rest dataset uses 62 channels.

## Corrections to the UniEQA sources

- **BCI Competition IV, data set 1.** Exclude the three records identified as artificial by the competition organizers. The official files encode the first listed class as −1 and the second as +1. UniEQA used the reverse mapping. Correlation with the official 100 Hz calibration cues found reversals in 200 of 200 windows for sub1 (left hand and foot), 200 of 200 for sub2 and 197 of 200 for sub7 (left and right hand). Under official labels, the 8–30 Hz share of power was lower over the hemisphere opposite the imagined hand, with z = 7.1 for sub7 and 1.0 for sub2. The sign reversed under the UniEQA labels. The benchmark restores the official mapping for all included participants. [Competition description](https://doi.org/10.3389/fnins.2012.00055)
- **VR neurofeedback.** The two rest periods last about 90 s each. For 33 of the 34 participants checked, UniEQA labels both periods as eyes open. For the remaining participant it labels the first period as eyes closed. The recorded order is open eyes followed by closed eyes. Relative posterior 8–13 Hz power was higher in the second period for 33 of 34 participants. The benchmark uses the recorded order and omits eye-state questions for the participant whose signal disagrees.
- **Photic stimulation.** UniEQA retains the stimulation label after the flashes stop. The benchmark aligns the question windows to the original flash markers and uses rest questions for windows without flashes.
- **WAY-EEG-GAL.** UniEQA uses the first lift's time window for almost every segment in a series. The benchmark cuts each lift from its own window in the original WS files. Applying the same window rule to the first lift reproduces 2,743 UniEQA segments sample by sample. The original surface codes are 1 for silk and 3 for sandpaper. The direction of the grip-to-load ratio agrees with the source codes. The benchmark exchanges the silk and sandpaper names and retains suede.
- **Exoskeleton.** The headers of 320 segments list 99.99999999999999 Hz. Truncating the rate to 99 Hz shifts the segment starts up to 1.46 s early. The benchmark reads these recordings at 100 Hz.
- **CHB-MIT.** One participant has 26 common-reference segments. The benchmark computes 18 bipolar channels by subtracting electrode pairs, which cancels the reference. For seizure segments, the median channel standard deviation was 42.5 µV after conversion and 40.6 µV for that participant's native bipolar recordings.

## Question eligibility

A training cell with a single answer class is omitted because the window length would reveal the answer. Scored tasks have at least 10 windows for each offered answer in each of validation and test. LEMON's 55–65 and 65–70 age bins were merged after the test label counts showed that the younger bin contained no test participant. Age and gender are constant across a participant's windows.
