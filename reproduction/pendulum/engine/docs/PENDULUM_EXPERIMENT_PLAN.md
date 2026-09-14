# Pendulum shortcut experiment plan

## 1. Task definition

This benchmark studies whether pendulum future prediction follows the true
physical history or appearance shortcuts.

The label families are:

- Appearance (`A`)
  - colour;
  - shape.
- Physics (`P`)
  - frequency;
  - amplitude.

The concrete label values are:

- Colour
  - training colours: `red`, `blue`;
  - OOD colours: `gray`, `green`.
- Shape
  - `circle`;
  - `square`.
- Frequency
  - `high_frequency`;
  - `low_frequency`.
- Amplitude
  - `large_amplitude`;
  - `small_amplitude`.

The pendulum length is fixed and is not a label or shortcut variable.

## 2. Complete generated data

The generator creates the full Cartesian product

```text
2 frequencies × 2 amplitudes × 4 colours × 2 shapes = 32 groups
```

using the name

```text
{frequency}-{amplitude}-{colour}-{shape}
```

### 2.1 High frequency and small amplitude

1. `high_frequency-small_amplitude-red-circle`
2. `high_frequency-small_amplitude-red-square`
3. `high_frequency-small_amplitude-blue-circle`
4. `high_frequency-small_amplitude-blue-square`
5. `high_frequency-small_amplitude-gray-circle`
6. `high_frequency-small_amplitude-gray-square`
7. `high_frequency-small_amplitude-green-circle`
8. `high_frequency-small_amplitude-green-square`

### 2.2 High frequency and large amplitude

9. `high_frequency-large_amplitude-red-circle`
10. `high_frequency-large_amplitude-red-square`
11. `high_frequency-large_amplitude-blue-circle`
12. `high_frequency-large_amplitude-blue-square`
13. `high_frequency-large_amplitude-gray-circle`
14. `high_frequency-large_amplitude-gray-square`
15. `high_frequency-large_amplitude-green-circle`
16. `high_frequency-large_amplitude-green-square`

### 2.3 Low frequency and small amplitude

17. `low_frequency-small_amplitude-red-circle`
18. `low_frequency-small_amplitude-red-square`
19. `low_frequency-small_amplitude-blue-circle`
20. `low_frequency-small_amplitude-blue-square`
21. `low_frequency-small_amplitude-gray-circle`
22. `low_frequency-small_amplitude-gray-square`
23. `low_frequency-small_amplitude-green-circle`
24. `low_frequency-small_amplitude-green-square`

### 2.4 Low frequency and large amplitude

25. `low_frequency-large_amplitude-red-circle`
26. `low_frequency-large_amplitude-red-square`
27. `low_frequency-large_amplitude-blue-circle`
28. `low_frequency-large_amplitude-blue-square`
29. `low_frequency-large_amplitude-gray-circle`
30. `low_frequency-large_amplitude-gray-square`
31. `low_frequency-large_amplitude-green-circle`
32. `low_frequency-large_amplitude-green-square`

Red and blue are available for training and testing. Gray and green are used
only for OOD testing.

All appearance variants associated with one `trajectory_id` must preserve the
same physical trajectory exactly.

## 3. Matched physics construction

### 3.1 Frequency target

Frequency experiments use `small_amplitude`.

For each base seed, the high- and low-frequency trajectories have:

- different sampled frequencies from their respective bands;
- exactly the same amplitude;
- exactly the same phase;
- the same pendulum length and renderer nuisance.

High and low frequency each account for 50% of every training or test set.

### 3.2 Amplitude target

Amplitude experiments use `low_frequency`.

For each base seed, the large- and small-amplitude trajectories have:

- different sampled amplitudes from their respective bands;
- exactly the same frequency;
- exactly the same phase;
- the same pendulum length and renderer nuisance.

Large and small amplitude each account for 50% of every training or test set.

## 4. Model training

Each logical model has independently trained short- and long-history versions.
They use the same raw physical trajectories and future frames, but different
history constructions.

### 4.1 Frequency models

#### `frequency_color_circle`

Training data:

- `high_frequency-small_amplitude-blue-circle`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- blue → high frequency;
- red → low frequency;
- shape is fixed to circle.

#### `frequency_color_square`

Training data:

- `high_frequency-small_amplitude-blue-square`;
- `low_frequency-small_amplitude-red-square`.

Binding:

- blue → high frequency;
- red → low frequency;
- shape is fixed to square.

#### `frequency_shape_red`

Training data:

- `high_frequency-small_amplitude-red-square`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- square → high frequency;
- circle → low frequency;
- colour is fixed to red.

#### `frequency_shape_blue`

Training data:

- `high_frequency-small_amplitude-blue-square`;
- `low_frequency-small_amplitude-blue-circle`.

Binding:

- square → high frequency;
- circle → low frequency;
- colour is fixed to blue.

#### `frequency_color_shape`

Training data:

- `high_frequency-small_amplitude-blue-square`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- blue + square → high frequency;
- red + circle → low frequency.

### 4.2 Amplitude models

#### `amplitude_color_circle`

Training data:

- `low_frequency-large_amplitude-blue-circle`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- blue → large amplitude;
- red → small amplitude;
- shape is fixed to circle.

#### `amplitude_color_square`

Training data:

- `low_frequency-large_amplitude-blue-square`;
- `low_frequency-small_amplitude-red-square`.

Binding:

- blue → large amplitude;
- red → small amplitude;
- shape is fixed to square.

#### `amplitude_shape_red`

Training data:

- `low_frequency-large_amplitude-red-square`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- square → large amplitude;
- circle → small amplitude;
- colour is fixed to red.

#### `amplitude_shape_blue`

Training data:

- `low_frequency-large_amplitude-blue-square`;
- `low_frequency-small_amplitude-blue-circle`.

Binding:

- square → large amplitude;
- circle → small amplitude;
- colour is fixed to blue.

#### `amplitude_color_shape`

Training data:

- `low_frequency-large_amplitude-blue-square`;
- `low_frequency-small_amplitude-red-circle`.

Binding:

- blue + square → large amplitude;
- red + circle → small amplitude.

## 5. Frequency experiments

In this plan, an ID-palette test uses the red/blue and circle/square values
seen during training while swapping their binding to the physical label. It is
an in-support intervention, not an unchanged training assignment.

### 5.1 Frequency–colour

Models:

1. `frequency_color_circle`;
2. `frequency_color_square`;
3. `frequency_color_shape`.

#### `frequency_color_circle`

Red/blue swap:

- `high_frequency-small_amplitude-red-circle`;
- `low_frequency-small_amplitude-blue-circle`.

Gray OOD:

- `high_frequency-small_amplitude-gray-circle`;
- `low_frequency-small_amplitude-gray-circle`.

Green OOD:

- `high_frequency-small_amplitude-green-circle`;
- `low_frequency-small_amplitude-green-circle`.

Shape remains fixed to circle in training and testing.

#### `frequency_color_square`

Red/blue swap:

- `high_frequency-small_amplitude-red-square`;
- `low_frequency-small_amplitude-blue-square`.

Gray OOD:

- `high_frequency-small_amplitude-gray-square`;
- `low_frequency-small_amplitude-gray-square`.

Green OOD:

- `high_frequency-small_amplitude-green-square`;
- `low_frequency-small_amplitude-green-square`.

Shape remains fixed to square in training and testing.

#### `frequency_color_shape`

Colour-only swap, preserving the trained shape binding:

- `high_frequency-small_amplitude-red-square`;
- `low_frequency-small_amplitude-blue-circle`.

Gray OOD, preserving the trained shape binding:

- `high_frequency-small_amplitude-gray-square`;
- `low_frequency-small_amplitude-gray-circle`.

Green OOD, preserving the trained shape binding:

- `high_frequency-small_amplitude-green-square`;
- `low_frequency-small_amplitude-green-circle`.

### 5.2 Frequency–shape

Models:

1. `frequency_shape_red`;
2. `frequency_shape_blue`;
3. `frequency_color_shape`.

No gray/green OOD evaluation is used in this experiment.

#### `frequency_shape_red`

Shape-only swap, keeping colour fixed:

- `high_frequency-small_amplitude-red-circle`;
- `low_frequency-small_amplitude-red-square`.

#### `frequency_shape_blue`

Shape-only swap, keeping colour fixed:

- `high_frequency-small_amplitude-blue-circle`;
- `low_frequency-small_amplitude-blue-square`.

#### `frequency_color_shape`

Shape-only swap, preserving the trained colour binding:

- `high_frequency-small_amplitude-blue-circle`;
- `low_frequency-small_amplitude-red-square`.

### 5.3 Frequency colour+shape joint swap

Model:

- `frequency_color_shape`.

Colour and shape are both swapped:

- `high_frequency-small_amplitude-red-circle`;
- `low_frequency-small_amplitude-blue-square`.

No gray/green OOD evaluation is used.

## 6. Amplitude experiments

### 6.1 Amplitude–colour

Models:

1. `amplitude_color_circle`;
2. `amplitude_color_square`;
3. `amplitude_color_shape`.

#### `amplitude_color_circle`

Red/blue swap:

- `low_frequency-large_amplitude-red-circle`;
- `low_frequency-small_amplitude-blue-circle`.

Gray OOD:

- `low_frequency-large_amplitude-gray-circle`;
- `low_frequency-small_amplitude-gray-circle`.

Green OOD:

- `low_frequency-large_amplitude-green-circle`;
- `low_frequency-small_amplitude-green-circle`.

#### `amplitude_color_square`

Red/blue swap:

- `low_frequency-large_amplitude-red-square`;
- `low_frequency-small_amplitude-blue-square`.

Gray OOD:

- `low_frequency-large_amplitude-gray-square`;
- `low_frequency-small_amplitude-gray-square`.

Green OOD:

- `low_frequency-large_amplitude-green-square`;
- `low_frequency-small_amplitude-green-square`.

#### `amplitude_color_shape`

Colour-only swap, preserving the trained shape binding:

- `low_frequency-large_amplitude-red-square`;
- `low_frequency-small_amplitude-blue-circle`.

Gray OOD, preserving the trained shape binding:

- `low_frequency-large_amplitude-gray-square`;
- `low_frequency-small_amplitude-gray-circle`.

Green OOD, preserving the trained shape binding:

- `low_frequency-large_amplitude-green-square`;
- `low_frequency-small_amplitude-green-circle`.

### 6.2 Amplitude–shape

Models:

1. `amplitude_shape_red`;
2. `amplitude_shape_blue`;
3. `amplitude_color_shape`.

No gray/green OOD evaluation is used in this experiment.

#### `amplitude_shape_red`

Shape-only swap, keeping colour fixed:

- `low_frequency-large_amplitude-red-circle`;
- `low_frequency-small_amplitude-red-square`.

#### `amplitude_shape_blue`

Shape-only swap, keeping colour fixed:

- `low_frequency-large_amplitude-blue-circle`;
- `low_frequency-small_amplitude-blue-square`.

#### `amplitude_color_shape`

Shape-only swap, preserving the trained colour binding:

- `low_frequency-large_amplitude-blue-circle`;
- `low_frequency-small_amplitude-red-square`.

### 6.3 Amplitude colour+shape joint swap

Model:

- `amplitude_color_shape`.

Colour and shape are both swapped:

- `low_frequency-large_amplitude-red-circle`;
- `low_frequency-small_amplitude-blue-square`.

No gray/green OOD evaluation is used.

## 7. Short/long prediction protocol

Short and long both predict the same 64 future pixel frames:

```text
future = frames 65..128
```

They differ only in the conditioning history:

- long condition: real pixel frames 0–64;
- short condition:
  - frames 0–56 are replaced with the same fixed background-only mask for
    every sample;
  - real pixel frames 57–64 remain visible.

For every short/long comparison:

- the raw 129-frame source video is shared;
- `trajectory_id`, frequency, amplitude and phase are identical;
- frames 57–128 are pixel-identical before encoding;
- the target future is always the same 64 pixel frames;
- the same train/eval sample IDs are used.

Short and long require independently trained checkpoints because their
conditioning distributions differ. This does **not** change output length:
both checkpoints generate 64 future frames.

The short checkpoint is evaluated with the short-history condition, and the
long checkpoint is evaluated with the long-history condition. Results must not
be substituted across history types, physical targets, or training bindings.

## 8. Result isolation

Every result records:

- physical target (`frequency` or `amplitude`);
- logical model name;
- history type (`short` or `long`);
- checkpoint/training ID;
- test intervention;
- test manifest ID.

Models trained with different bindings are independent. In particular:

- a `color_shape` result is not a `color`-only model result;
- a colour-swap result is not a shape-swap result;
- a frequency model is not reused for amplitude;
- short and long use their corresponding independently trained checkpoints,
  although both generate the same 64-frame future.
